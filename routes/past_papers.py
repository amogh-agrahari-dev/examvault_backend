import os
import uuid
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel

from database import get_db
from models.auth_models import User
from models.subject_models import Subject
from models.document_models import Document, Topic
from models.chat_models import DocumentChunk
from models.past_paper_models import PastPaper, PredictedExamPaper
from routes.auth import get_current_user
from services.document_parser import extract_text, extract_past_paper_content
from services.paper_prediction_service import predict_exam_paper_from_past_paper

logger = logging.getLogger(__name__)

# Dedicated directory strictly separated from general notes/documents
PAST_PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads", "past_papers")
os.makedirs(PAST_PAPERS_DIR, exist_ok=True)

router = APIRouter(prefix="/api/v1/past-papers", tags=["past_papers"])

class PredictPaperRequest(BaseModel):
    target_year: Optional[int] = 2025
    total_marks: Optional[int] = 70
    difficulty: Optional[str] = "medium"
    subject_name: Optional[str] = None

class ImportFromDocumentRequest(BaseModel):
    document_id: uuid.UUID
    year: Optional[int] = None
    exam_type: Optional[str] = "End-Sem"

@router.post("/upload")
async def upload_past_paper(
    file: UploadFile = File(...),
    subject_id: Optional[str] = Form(None),
    year: Optional[str] = Form(None),
    exam_type: Optional[str] = Form("End-Sem"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Uploads a previous year examination paper into the dedicated past_papers filebase.
    Never merges or pollutes the main documents filebase.
    """
    try:
        subject_uuid = uuid.UUID(subject_id) if subject_id else None
    except (ValueError, TypeError):
        subject_uuid = None

    try:
        year_int = int(year) if year else None
    except (ValueError, TypeError):
        year_int = None

    paper_id = uuid.uuid4()
    clean_filename = os.path.basename(file.filename or f"past_paper_{paper_id}.pdf")
    local_path = os.path.join(PAST_PAPERS_DIR, f"{paper_id}_{clean_filename}")

    try:
        content = await file.read()
        with open(local_path, "wb") as f:
            f.write(content)
        file_size = len(content)
        mime_type = file.content_type or "application/pdf"

        # Extract text directly from the past paper (supports digital and scanned hardcopies via vision OCR)
        extracted_text = await extract_past_paper_content(local_path, mime_type)
        if not extracted_text or len(extracted_text.strip()) < 10:
            extracted_text = f"Examination Paper: {clean_filename}"

        past_paper = PastPaper(
            id=paper_id,
            user_id=current_user.id,
            subject_id=subject_uuid,
            filename=clean_filename,
            file_path=local_path,
            file_size=file_size,
            mime_type=mime_type,
            year=year_int,
            exam_type=exam_type or "End-Sem",
            extracted_text=extracted_text,
            status="processed"
        )
        db.add(past_paper)
        await db.commit()
        await db.refresh(past_paper)

        return {
            "id": str(past_paper.id),
            "filename": past_paper.filename,
            "file_size": past_paper.file_size,
            "year": past_paper.year,
            "exam_type": past_paper.exam_type,
            "subject_id": str(past_paper.subject_id) if past_paper.subject_id else None,
            "status": past_paper.status,
            "created_at": past_paper.created_at.isoformat() if past_paper.created_at else None,
            "text_length": len(past_paper.extracted_text or "")
        }
    except Exception as e:
        logger.error(f"Failed to upload and parse past paper: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process past paper: {str(e)}")


@router.post("/import-from-document")
async def import_from_main_document(
    req: ImportFromDocumentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Imports a file already uploaded to the main filebase (documents table)
    to become a past paper / exam resource in the dedicated paper prediction filebase.
    """
    doc = await db.scalar(
        select(Document).where(Document.id == req.document_id, Document.user_id == current_user.id)
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found in main filebase")

    paper_id = uuid.uuid4()
    clean_filename = os.path.basename(doc.filename or f"document_{doc.id}.pdf")
    destination_path = os.path.join(PAST_PAPERS_DIR, f"{paper_id}_{clean_filename}")

    # 1. Retrieve text from DocumentChunk if already processed for instant, pristine fidelity
    extracted_text = ""
    chunks_res = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc.id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    chunks = chunks_res.scalars().all()
    if chunks:
        extracted_text = "\n\n".join(c.content for c in chunks if c.content and c.content.strip())
        logger.info(f"Retrieved {len(chunks)} chunks ({len(extracted_text)} chars) from main document {doc.id}")

    # 2. Check main uploads directory for the physical file
    main_upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
    source_file_path = os.path.join(main_upload_dir, f"{doc.id}_{clean_filename}")
    if not os.path.exists(source_file_path) and os.path.exists(main_upload_dir):
        for f in os.listdir(main_upload_dir):
            if f.startswith(str(doc.id)):
                source_file_path = os.path.join(main_upload_dir, f)
                break

    if os.path.exists(source_file_path):
        try:
            with open(source_file_path, "rb") as sf:
                file_bytes = sf.read()
            with open(destination_path, "wb") as df:
                df.write(file_bytes)
        except Exception as copy_err:
            logger.warning(f"Failed to copy main file {source_file_path} to past_papers: {copy_err}")

    # If text was not in chunks and local file exists, extract text from file
    if (not extracted_text or len(extracted_text.strip()) < 20) and os.path.exists(destination_path):
        extracted_text = await extract_past_paper_content(destination_path, doc.mime_type)

    # If still not found and appwrite_file_id exists, download from Appwrite
    if (not extracted_text or len(extracted_text.strip()) < 20) and doc.appwrite_file_id:
        try:
            from services.appwrite_service import download_file_from_appwrite
            await asyncio.to_thread(download_file_from_appwrite, doc.appwrite_file_id, destination_path)
            extracted_text = await extract_past_paper_content(destination_path, doc.mime_type)
        except Exception as aw_err:
            logger.warning(f"Failed to download from Appwrite during document import: {aw_err}")

    # If text is still empty, fallback to topic summaries
    if not extracted_text or len(extracted_text.strip()) < 20:
        topics_res = await db.execute(
            select(Topic)
            .where(Topic.document_id == doc.id)
            .order_by(Topic.order_index.asc())
        )
        topics = topics_res.scalars().all()
        if topics:
            extracted_text = "\n\n".join(
                f"### {t.name}\n{t.summary or ''}" for t in topics if t.summary
            )

    if not extracted_text or len(extracted_text.strip()) < 10:
        extracted_text = f"Examination Resource Document: {clean_filename}"

    # Ensure destination file exists on disk
    if not os.path.exists(destination_path):
        try:
            with open(destination_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
        except Exception as write_err:
            logger.warning(f"Could not write destination file: {write_err}")

    file_size = os.path.getsize(destination_path) if os.path.exists(destination_path) else (doc.file_size or len(extracted_text.encode("utf-8")))

    # Check if this document was previously imported
    existing = await db.scalar(
        select(PastPaper).where(
            PastPaper.user_id == current_user.id,
            PastPaper.filename == clean_filename
        )
    )
    if existing:
        existing.extracted_text = extracted_text
        if req.year:
            existing.year = req.year
        if req.exam_type:
            existing.exam_type = req.exam_type
        if doc.subject_id:
            existing.subject_id = doc.subject_id
        await db.commit()
        await db.refresh(existing)
        target_paper = existing
    else:
        past_paper = PastPaper(
            id=paper_id,
            user_id=current_user.id,
            subject_id=doc.subject_id,
            filename=clean_filename,
            file_path=destination_path,
            file_size=file_size,
            mime_type=doc.mime_type or "application/pdf",
            year=req.year or doc.year or (datetime.now(timezone.utc).year - 1),
            exam_type=req.exam_type or "End-Sem",
            extracted_text=extracted_text,
            status="processed"
        )
        db.add(past_paper)
        await db.commit()
        await db.refresh(past_paper)
        target_paper = past_paper

    return {
        "id": str(target_paper.id),
        "filename": target_paper.filename,
        "file_size": target_paper.file_size,
        "year": target_paper.year,
        "exam_type": target_paper.exam_type,
        "subject_id": str(target_paper.subject_id) if target_paper.subject_id else None,
        "status": target_paper.status,
        "created_at": target_paper.created_at.isoformat() if target_paper.created_at else None,
        "text_length": len(target_paper.extracted_text or ""),
        "source_document_id": str(doc.id)
    }


@router.get("/")
async def list_past_papers(
    subject_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Lists past examination papers in the dedicated filebase for the current user.
    """
    query = select(PastPaper).where(PastPaper.user_id == current_user.id)
    if subject_id:
        try:
            s_uuid = uuid.UUID(subject_id)
            query = query.where(PastPaper.subject_id == s_uuid)
        except (ValueError, TypeError):
            pass

    query = query.order_by(PastPaper.created_at.desc())
    result = await db.execute(query)
    papers = result.scalars().all()

    # Get prediction counts per paper
    response_items = []
    for p in papers:
        pred_count = await db.scalar(
            select(func.count(PredictedExamPaper.id)).where(PredictedExamPaper.past_paper_id == p.id)
        )
        response_items.append({
            "id": str(p.id),
            "filename": p.filename,
            "file_size": p.file_size,
            "year": p.year,
            "exam_type": p.exam_type,
            "subject_id": str(p.subject_id) if p.subject_id else None,
            "status": p.status,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "text_preview": (p.extracted_text[:250] + "...") if p.extracted_text else "",
            "prediction_count": pred_count or 0
        })

    return {"past_papers": response_items}


@router.get("/{paper_id}")
async def get_past_paper(
    paper_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    paper = await db.scalar(
        select(PastPaper).where(PastPaper.id == paper_id, PastPaper.user_id == current_user.id)
    )
    if not paper:
        raise HTTPException(status_code=404, detail="Past paper not found")

    return {
        "id": str(paper.id),
        "filename": paper.filename,
        "file_size": paper.file_size,
        "year": paper.year,
        "exam_type": paper.exam_type,
        "subject_id": str(paper.subject_id) if paper.subject_id else None,
        "extracted_text": paper.extracted_text,
        "created_at": paper.created_at.isoformat() if paper.created_at else None
    }


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_past_paper(
    paper_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    paper = await db.scalar(
        select(PastPaper).where(PastPaper.id == paper_id, PastPaper.user_id == current_user.id)
    )
    if not paper:
        raise HTTPException(status_code=404, detail="Past paper not found")

    # Remove local file if exists
    if paper.file_path and os.path.exists(paper.file_path):
        try:
            os.remove(paper.file_path)
        except Exception as e:
            logger.warning(f"Failed to remove past paper file on disk: {e}")

    await db.delete(paper)
    await db.commit()


@router.post("/{paper_id}/predict")
async def predict_paper_endpoint(
    paper_id: uuid.UUID,
    req: PredictPaperRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Predicts upcoming examination questions INDEPENDENTLY and SOLELY
    based on the uploaded previous year paper.
    Does NOT navigate or touch database topics/chunks.
    """
    paper = await db.scalar(
        select(PastPaper).where(PastPaper.id == paper_id, PastPaper.user_id == current_user.id)
    )
    if not paper:
        raise HTTPException(status_code=404, detail="Past paper not found")

    if not paper.extracted_text or len(paper.extracted_text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="The uploaded past paper has insufficient text for prediction. Please upload a clear text or PDF paper."
        )

    # Determine subject name from subject relation if available
    subject_name = req.subject_name
    if not subject_name and paper.subject_id:
        sub = await db.get(Subject, paper.subject_id)
        if sub:
            subject_name = sub.name

    prediction_result = await predict_exam_paper_from_past_paper(
        past_paper_text=paper.extracted_text,
        subject_name=subject_name,
        target_year=req.target_year or 2025,
        total_marks=req.total_marks or 70,
        difficulty=req.difficulty or "medium"
    )

    pred_paper = PredictedExamPaper(
        id=uuid.uuid4(),
        user_id=current_user.id,
        subject_id=paper.subject_id,
        past_paper_id=paper.id,
        title=prediction_result.get("exam_title", f"Predicted Examination Paper - {req.target_year or 2025}"),
        subject_name=prediction_result.get("subject_name", subject_name or "Academic Examination"),
        target_year=prediction_result.get("target_year", req.target_year or 2025),
        total_marks=prediction_result.get("total_marks", req.total_marks or 70),
        duration_minutes=prediction_result.get("duration_minutes", 180),
        difficulty=req.difficulty or "medium",
        general_instructions=prediction_result.get("general_instructions", []),
        sections=prediction_result.get("sections", []),
        marking_scheme=prediction_result.get("marking_scheme", {}),
        summary_analysis=prediction_result.get("summary_analysis", "")
    )
    db.add(pred_paper)
    await db.commit()
    await db.refresh(pred_paper)

    return {
        "id": str(pred_paper.id),
        "title": pred_paper.title,
        "subject_name": pred_paper.subject_name,
        "target_year": pred_paper.target_year,
        "total_marks": pred_paper.total_marks,
        "duration_minutes": pred_paper.duration_minutes,
        "difficulty": pred_paper.difficulty,
        "general_instructions": pred_paper.general_instructions,
        "summary_analysis": pred_paper.summary_analysis,
        "sections": pred_paper.sections,
        "created_at": pred_paper.created_at.isoformat() if pred_paper.created_at else None
    }


@router.get("/predicted/list")
async def list_predicted_papers(
    subject_id: Optional[str] = None,
    past_paper_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Lists all predicted exam papers generated for the current user."""
    query = select(PredictedExamPaper).where(PredictedExamPaper.user_id == current_user.id)
    if subject_id:
        try:
            s_uuid = uuid.UUID(subject_id)
            query = query.where(PredictedExamPaper.subject_id == s_uuid)
        except (ValueError, TypeError):
            pass
    if past_paper_id:
        try:
            p_uuid = uuid.UUID(past_paper_id)
            query = query.where(PredictedExamPaper.past_paper_id == p_uuid)
        except (ValueError, TypeError):
            pass

    query = query.order_by(PredictedExamPaper.created_at.desc())
    result = await db.execute(query)
    papers = result.scalars().all()

    return {
        "predicted_papers": [
            {
                "id": str(p.id),
                "title": p.title,
                "subject_name": p.subject_name,
                "target_year": p.target_year,
                "total_marks": p.total_marks,
                "duration_minutes": p.duration_minutes,
                "difficulty": p.difficulty,
                "past_paper_id": str(p.past_paper_id) if p.past_paper_id else None,
                "subject_id": str(p.subject_id) if p.subject_id else None,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "section_count": len(p.sections or [])
            }
            for p in papers
        ]
    }


@router.get("/predicted/{predicted_id}")
async def get_predicted_paper(
    predicted_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    paper = await db.scalar(
        select(PredictedExamPaper).where(
            PredictedExamPaper.id == predicted_id,
            PredictedExamPaper.user_id == current_user.id
        )
    )
    if not paper:
        raise HTTPException(status_code=404, detail="Predicted paper not found")

    return {
        "id": str(paper.id),
        "title": paper.title,
        "subject_name": paper.subject_name,
        "target_year": paper.target_year,
        "total_marks": paper.total_marks,
        "duration_minutes": paper.duration_minutes,
        "difficulty": paper.difficulty,
        "general_instructions": paper.general_instructions,
        "summary_analysis": paper.summary_analysis,
        "sections": paper.sections,
        "created_at": paper.created_at.isoformat() if paper.created_at else None
    }


@router.delete("/predicted/{predicted_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_predicted_paper(
    predicted_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    paper = await db.scalar(
        select(PredictedExamPaper).where(
            PredictedExamPaper.id == predicted_id,
            PredictedExamPaper.user_id == current_user.id
        )
    )
    if not paper:
        raise HTTPException(status_code=404, detail="Predicted paper not found")

    await db.delete(paper)
    await db.commit()
