import os
import uuid
import tempfile
import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from database import get_db
from routes.auth import get_current_user
from models.auth_models import User
from models.document_models import Document, Topic
from models.summary_models import SummaryJob, Summary
from services.summary_service import organize_topics_with_llm, extract_relevant_facts, generate_final_topic_summary
from services.appwrite_service import download_file_from_appwrite
from services.document_parser import extract_text, chunk_text

router = APIRouter(prefix="/api/v1/summary", tags=["summary"])

class DiscoverRequest(BaseModel):
    document_ids: List[str]

class GenerateRequest(BaseModel):
    selected_topics: dict

@router.post("/discover")
async def discover_topics(
    req: DiscoverRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        doc_uuids = [uuid.UUID(d) for d in req.document_ids]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    # Fetch topics for these docs
    result = await db.execute(
        select(Topic).where(Topic.document_id.in_(doc_uuids), Topic.user_id == current_user.id)
    )
    topics = result.scalars().all()
    
    if not topics:
        raise HTTPException(status_code=404, detail="No topics found for the selected documents. Please ensure documents have been processed.")

    topics_data = [{"id": str(t.id), "name": t.name, "summary": t.summary} for t in topics]
    
    # Run Ollama to organize topics
    organized_tree = await organize_topics_with_llm(topics_data)
    
    # Create Job
    job = SummaryJob(
        user_id=current_user.id,
        selected_documents={"ids": req.document_ids},
        discovered_topics=organized_tree,
        status="discovered"
    )
    db.add(job)
    await db.commit()
    
    return {"job_id": str(job.id), "discovered_topics": organized_tree}


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(SummaryJob).where(SummaryJob.id == uuid.UUID(job_id), SummaryJob.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return {
        "id": str(job.id),
        "status": job.status,
        "progress": job.progress,
        "current_stage": job.current_stage,
        "error_details": job.error_details,
        "discovered_topics": job.discovered_topics,
        "selected_topics": job.selected_topics
    }


@router.post("/jobs/{job_id}/generate")
async def start_generation(
    job_id: str,
    req: GenerateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(SummaryJob).where(SummaryJob.id == uuid.UUID(job_id), SummaryJob.user_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job.selected_topics = req.selected_topics
    job.status = "generating"
    job.progress = 0
    
    # Reward 5 Vault credits for generating a summary
    from sqlalchemy import update
    from services.redis_service import invalidate_user_cache
    await db.execute(
        update(User)
        .where(User.id == current_user.id)
        .values(reputation_credits=User.reputation_credits + 5)
    )
    await db.commit()
    await invalidate_user_cache(str(current_user.id))
    current_user.reputation_credits += 5
    
    background_tasks.add_task(generate_summaries_task, job.id, current_user.id)
    
    return {"status": "started", "job_id": str(job.id), "credits": current_user.reputation_credits}


async def generate_summaries_task(job_id: uuid.UUID, user_id: uuid.UUID):
    # This must create its own DB session since it runs in background
    from database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(SummaryJob).where(SummaryJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return
            
        try:
            # 1. Fetch documents
            doc_ids = [uuid.UUID(d) for d in job.selected_documents["ids"]]
            result = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
            docs = result.scalars().all()
            
            # Download and chunk documents
            all_chunks = []
            for d in docs:
                if not d.appwrite_file_id:
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(d.filename)[1]) as tmp:
                    temp_path = tmp.name
                
                await asyncio.to_thread(download_file_from_appwrite, d.appwrite_file_id, temp_path)
                text = await asyncio.to_thread(extract_text, temp_path, d.mime_type)
                chunks = chunk_text(text, max_words=1000)
                all_chunks.extend(chunks)
                os.remove(temp_path)
            
            tree = job.selected_topics.get("tree", [])
            total_topics = len(tree)
            
            for i, topic_node in enumerate(tree):
                job.current_stage = f"Generating summary for {topic_node['name']}"
                await db.commit()
                
                # Extract facts across all chunks
                intermediate_notes = ""
                for chunk in all_chunks:
                    facts = await extract_relevant_facts(chunk, topic_node["name"])
                    if facts:
                        intermediate_notes += facts + "\n\n"
                        
                if not intermediate_notes.strip():
                    intermediate_notes = "No specific details found in the selected documents."
                    
                # Generate final summary
                final_summary = await generate_final_topic_summary(
                    topic_node["name"],
                    topic_node.get("subtopics", []),
                    intermediate_notes
                )
                
                # Save Summary
                summary = Summary(
                    user_id=user_id,
                    job_id=job.id,
                    topic_name=topic_node["name"],
                    content=final_summary,
                    source_documents=job.selected_documents
                )
                db.add(summary)
                
                job.progress = int(((i + 1) / total_topics) * 100)
                await db.commit()
                
            job.status = "completed"
            job.current_stage = "All summaries generated successfully"
            await db.commit()
            
        except Exception as e:
            job.status = "failed"
            job.error_details = str(e)
            await db.commit()
            import traceback
            traceback.print_exc()

@router.get("/")
async def get_summaries(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Summary).where(Summary.user_id == current_user.id).order_by(Summary.created_at.desc())
    )
    summaries = result.scalars().all()
    return {"summaries": summaries}

@router.get("/{summary_id}")
async def get_summary(
    summary_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Summary).where(Summary.id == uuid.UUID(summary_id), Summary.user_id == current_user.id)
    )
    summary = result.scalar_one_or_none()
    if not summary:
        raise HTTPException(status_code=404, detail="Summary not found")
    return {"summary": summary}
