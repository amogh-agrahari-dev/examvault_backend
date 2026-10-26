import uuid
import random
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from database import get_db
from models.auth_models import User
from models.subject_models import Subject, PastPaperQuestion, ExtractedConcept
from models.document_models import Document
from models.quiz_models import Quiz, QuizQuestion
from models.chat_models import DocumentChunk
from routes.auth import get_current_user
from services.ollama_service import extract_past_paper_questions, generate_new_question_from_context
from services.document_parser import extract_text, chunk_text
import tempfile
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/subjects", tags=["predicted_papers"])

class GeneratePaperRequest(BaseModel):
    total_marks: int = 50
    num_questions: int = 10
    difficulty: str = "medium"

@router.post("/{subject_id}/analyze")
async def analyze_past_papers(
    subject_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Triggers analysis on all PAST_PAPER documents for a subject that haven't been analyzed.
    For simplicity in this prototype, we'll run it synchronously or just queue it.
    """
    subject = await db.get(Subject, subject_id)
    if not subject or subject.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Subject not found")

    background_tasks.add_task(run_analysis_pipeline, subject_id, db)
    return {"message": "Analysis started in background"}

async def run_analysis_pipeline(subject_id: uuid.UUID, db: AsyncSession):
    # Get past papers
    result = await db.execute(
        select(Document).where(Document.subject_id == subject_id, Document.document_type == "PAST_PAPER")
    )
    docs = result.scalars().all()
    
    # In a real app, we'd fetch the file from Appwrite, extract text, chunk, and process.
    # We will simulate the extraction here using existing chunks if they exist, or just skip Appwrite download for the prototype and assume chunks exist.
    # Actually, process_document_task creates DocumentChunks. So we can just query DocumentChunk for these documents.
    for doc in docs:
        chunks_result = await db.execute(select(DocumentChunk).where(DocumentChunk.document_id == doc.id))
        chunks = chunks_result.scalars().all()
        
        for chunk in chunks:
            extracted = await extract_past_paper_questions(chunk.page_content)
            for q_data in extracted:
                concept_name = q_data.get("concept", "General").title()
                
                # Check if concept exists
                c_result = await db.execute(
                    select(ExtractedConcept).where(ExtractedConcept.subject_id == subject_id, ExtractedConcept.name == concept_name)
                )
                concept = c_result.scalar_one_or_none()
                
                if not concept:
                    concept = ExtractedConcept(
                        subject_id=subject_id,
                        name=concept_name,
                        frequency=1,
                        importance_score=1.0
                    )
                    db.add(concept)
                    await db.commit()
                    await db.refresh(concept)
                else:
                    concept.frequency += 1
                    concept.importance_score += 1.0
                    await db.commit()
                
                # Save Question
                pq = PastPaperQuestion(
                    subject_id=subject_id,
                    document_id=doc.id,
                    concept_id=concept.id,
                    original_text=q_data.get("original_text", ""),
                    year=doc.year,
                    marks=q_data.get("marks", 5),
                    question_type=q_data.get("question_type", "short"),
                    difficulty=q_data.get("difficulty", "medium")
                )
                db.add(pq)
                await db.commit()

@router.get("/{subject_id}/analysis")
async def get_analysis_dashboard(
    subject_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    subject = await db.get(Subject, subject_id)
    if not subject or subject.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Subject not found")
        
    result = await db.execute(
        select(ExtractedConcept)
        .where(ExtractedConcept.subject_id == subject_id)
        .order_by(ExtractedConcept.importance_score.desc())
        .limit(10)
    )
    concepts = result.scalars().all()
    
    return [
        {
            "id": c.id,
            "name": c.name,
            "frequency": c.frequency,
            "importance_score": c.importance_score
        } for c in concepts
    ]

@router.post("/{subject_id}/generate-paper")
async def generate_predicted_paper(
    subject_id: uuid.UUID,
    req: GeneratePaperRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    subject = await db.get(Subject, subject_id)
    if not subject or subject.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Subject not found")

    # 1. Get Top Concepts
    c_res = await db.execute(
        select(ExtractedConcept)
        .where(ExtractedConcept.subject_id == subject_id)
        .order_by(ExtractedConcept.importance_score.desc())
    )
    concepts = c_res.scalars().all()
    
    if not concepts:
        raise HTTPException(status_code=400, detail="No past paper analysis found. Analyze past papers first.")

    # 2. Split 50/50
    historical_count = req.num_questions // 2
    new_count = req.num_questions - historical_count
    
    paper_questions = []
    
    # Select historical questions
    for i in range(min(historical_count, len(concepts))):
        concept = concepts[i]
        q_res = await db.execute(
            select(PastPaperQuestion)
            .where(PastPaperQuestion.concept_id == concept.id)
            .limit(1)
        )
        pq = q_res.scalar_one_or_none()
        if pq:
            paper_questions.append({
                "question_text": pq.original_text,
                "suggested_answer": "Review past paper material for this answer.",
                "marks": pq.marks or 5,
                "difficulty": pq.difficulty or "medium",
                "source": "historical"
            })
            
    # Generate new questions for the remaining concepts
    # Get NOTES chunks
    n_res = await db.execute(
        select(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.subject_id == subject_id, Document.document_type == "NOTES")
        .limit(50)
    )
    notes_chunks = n_res.scalars().all()
    context_text = "\n".join([nc.page_content for nc in notes_chunks]) if notes_chunks else "General knowledge."

    for i in range(new_count):
        concept = concepts[(historical_count + i) % len(concepts)]
        new_q = await generate_new_question_from_context(
            concept=concept.name,
            context=context_text[:2000], # Pass a limited context
            marks=req.total_marks // req.num_questions,
            difficulty=req.difficulty
        )
        
        if new_q and "question_text" in new_q:
            paper_questions.append({
                "question_text": new_q.get("question_text"),
                "suggested_answer": new_q.get("suggested_answer", ""),
                "marks": new_q.get("marks", 5),
                "difficulty": req.difficulty,
                "source": "generated"
            })
            
    # Save as Quiz
    quiz = Quiz(
        user_id=current_user.id,
        title=f"Predicted Paper: {subject.name}",
        selected_documents=[],
        num_questions=len(paper_questions),
        difficulty_distribution={"medium": 1.0}
    )
    db.add(quiz)
    await db.commit()
    await db.refresh(quiz)
    
    for i, q in enumerate(paper_questions):
        qq = QuizQuestion(
            quiz_id=quiz.id,
            question_text=q["question_text"],
            options=[],
            correct_answer=q["suggested_answer"],
            explanation=f"Source: {q['source']}. Marks: {q['marks']}",
            difficulty=q["difficulty"],
            order_index=i
        )
        db.add(qq)
        
    await db.commit()
    
    return {
        "quiz_id": quiz.id,
        "title": quiz.title,
        "questions": paper_questions
    }
