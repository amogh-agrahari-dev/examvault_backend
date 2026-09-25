import os
import uuid
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from database import get_db
from routes.auth import get_current_user
from models.auth_models import User
from models.document_models import Document, Topic, Flashcard
from services.appwrite_service import upload_file_to_appwrite
from services.document_parser import extract_text, chunk_text
from services.ollama_service import generate_flashcards
from services.flashcard_service import save_document, save_generated_content

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        file_size = os.path.getsize(temp_path)
        mime_type = file.content_type
        
        # 1. Upload to Appwrite
        appwrite_file_id = upload_file_to_appwrite(temp_path, file.filename)
        
        # 2. Save Document record
        doc = await save_document(
            db=db,
            user_id=current_user.id,
            appwrite_file_id=appwrite_file_id,
            filename=file.filename,
            mime_type=mime_type,
            file_size=file_size,
            status="processing"
        )

        # 3. Process the document
        background_tasks.add_task(
            process_document_task,
            db=db,
            document_id=doc.id,
            temp_path=temp_path,
            mime_type=mime_type,
            user_id=current_user.id
        )
        
        return {"document": {"id": str(doc.id), "filename": doc.filename, "status": doc.status}}
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_task(db: AsyncSession, document_id: uuid.UUID, temp_path: str, mime_type: str, user_id: uuid.UUID):
    try:
        # Re-fetch document
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return
            
        # Extract text
        text = extract_text(temp_path, mime_type)
        
        # Chunk text for large documents (1000 words per chunk for local models)
        chunks = chunk_text(text, max_words=1000)
        
        async def update_progress(current: int, total: int):
            try:
                res = await db.execute(select(Document).where(Document.id == document_id))
                doc_record = res.scalar_one_or_none()
                if doc_record:
                    doc_record.status = f"Processing chunk {current}/{total}"
                    await db.commit()
            except Exception as e:
                print(f"Error updating progress: {e}")
        
        # Generate with Ollama iteratively
        generated_data = await generate_flashcards(chunks, progress_callback=update_progress)
        
        # Save generated topics and flashcards
        await save_generated_content(db, document_id, user_id, generated_data)
        
        doc.status = "completed"
        await db.commit()
        
    except Exception as e:
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()
        if doc:
            doc.status = "failed"
            await db.commit()
        print(f"Error processing document {document_id}: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/")
async def get_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return {"documents": docs}


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        doc_uuid = uuid.UUID(document_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Document not found")

    result = await db.execute(
        select(Document)
        .where(Document.id == doc_uuid, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"document": doc}


@router.get("/{document_id}/topics")
async def get_document_topics(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        doc_uuid = uuid.UUID(document_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Document not found")

    # Verify document ownership
    result = await db.execute(select(Document).where(Document.id == doc_uuid, Document.user_id == current_user.id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found")
        
    result = await db.execute(
        select(Topic)
        .where(Topic.document_id == doc_uuid)
        .order_by(Topic.order_index.asc())
    )
    topics = result.scalars().all()
    return {"topics": topics}


@router.get("/topics/{topic_id}/flashcards")
async def get_topic_flashcards(
    topic_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        topic_uuid = uuid.UUID(topic_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Topic not found")

    # Verify topic ownership
    result = await db.execute(select(Topic).where(Topic.id == topic_uuid, Topic.user_id == current_user.id))
    topic = result.scalar_one_or_none()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
        
    result = await db.execute(
        select(Flashcard)
        .where(Flashcard.topic_id == topic_uuid)
        .order_by(Flashcard.order_index.asc())
    )
    flashcards = result.scalars().all()
    return {"topic": topic, "flashcards": flashcards}
