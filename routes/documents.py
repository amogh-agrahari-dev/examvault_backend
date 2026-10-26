import os
import uuid
import tempfile
import asyncio
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from database import get_db, AsyncSessionLocal
from routes.auth import get_current_user
from models.auth_models import User
from models.document_models import Document, Topic, Flashcard
from services.appwrite_service import upload_file_to_appwrite, download_file_from_appwrite, delete_file_from_appwrite
from services.document_parser import extract_text, extract_document_content_and_images, chunk_text
from services.vision_service import EXTRACTED_IMAGES_DIR
from services.llm_router import generate_flashcards
from services.flashcard_service import save_document, save_generated_content
from services.redis_service import (
    get_cached_doc_topics, set_cached_doc_topics,
    get_cached_topic_flashcards, set_cached_topic_flashcards,
    get_cached_user_documents, set_cached_user_documents,
    invalidate_user_documents
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

@router.api_route("/images/{image_name}", methods=["GET", "HEAD"])
async def get_document_image(image_name: str):
    """Serves extracted document figures and diagrams to frontend flashcards and study views."""
    clean_name = os.path.basename(image_name)
    file_path = os.path.join(EXTRACTED_IMAGES_DIR, clean_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject_id: str = Form(None),
    document_type: str = Form("NOTES"),
    year: str = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Convert types if present
        subject_uuid = uuid.UUID(subject_id) if subject_id else None
        year_int = int(year) if year else None
        doc_id = uuid.uuid4()
        
        # Save file to server's uploads directory for reliable processing and local persistence
        clean_filename = os.path.basename(file.filename)
        local_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{clean_filename}")
        
        content = await file.read()
        with open(local_path, "wb") as f:
            f.write(content)
        
        file_size = len(content)
        mime_type = file.content_type
        
        # Non-blocking Appwrite upload (non-fatal if Appwrite rejects extension or is offline)
        appwrite_file_id = None
        try:
            appwrite_file_id = await asyncio.to_thread(upload_file_to_appwrite, local_path, clean_filename)
        except Exception as appwrite_err:
            print(f"Appwrite storage upload skipped/failed for {clean_filename}: {appwrite_err}")
        
        # Save Document record
        doc = Document(
            id=doc_id,
            user_id=current_user.id,
            appwrite_file_id=appwrite_file_id,
            filename=clean_filename,
            mime_type=mime_type,
            file_size=file_size,
            status="processing",
            subject_id=subject_uuid,
            document_type=document_type,
            year=year_int
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        # Process the document in background with its own independent DB session
        background_tasks.add_task(
            process_document_task,
            document_id=doc.id,
            file_path=local_path,
            mime_type=mime_type,
            user_id=current_user.id
        )
        
        # Invalidate documents cache for this user
        await invalidate_user_documents(str(current_user.id))

        # Reward 20 Vault credits for adding a document
        from sqlalchemy import update
        from services.redis_service import invalidate_user_cache
        await db.execute(
            update(User)
            .where(User.id == current_user.id)
            .values(reputation_credits=User.reputation_credits + 20)
        )
        await db.commit()
        await invalidate_user_cache(str(current_user.id))
        current_user.reputation_credits += 20
        
        return {"document": {"id": str(doc.id), "filename": doc.filename, "status": doc.status}}
        
    except Exception as e:
        print(f"Error in upload_document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_task(document_id: uuid.UUID, file_path: str, mime_type: str, user_id: uuid.UUID):
    """Background task with isolated session to prevent closed-session race conditions."""
    async with AsyncSessionLocal() as db:
        try:
            # Re-fetch document
            result = await db.execute(select(Document).where(Document.id == document_id))
            doc = result.scalar_one_or_none()
            if not doc:
                return
                
            # If local file does not exist, attempt to restore it from Appwrite
            if not os.path.exists(file_path):
                if doc.appwrite_file_id:
                    await asyncio.to_thread(download_file_from_appwrite, doc.appwrite_file_id, file_path)
                else:
                    raise FileNotFoundError(f"Document file {file_path} not found locally or in Appwrite")

            # Extract text AND embedded figures / diagrams via vision analysis
            text, extracted_images = await extract_document_content_and_images(
                file_path, mime_type or doc.mime_type, doc_id=document_id
            )
            if not text or not text.strip():
                raise ValueError("No readable text found in document.")
            
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
            
            # Generate with Ollama using parallel execution and Redis chunk caching
            generated_data = await generate_flashcards(chunks, progress_callback=update_progress)
            
            # Save generated topics, flashcards, and document chunks (for RAG chat)
            await save_generated_content(db, document_id, user_id, generated_data, raw_chunks=chunks)
            
            doc.status = "completed"
            await db.commit()

            # Invalidate document caches
            await invalidate_user_documents(str(user_id))
            
        except Exception as e:
            print(f"Error processing document {document_id}: {e}")
            import traceback
            traceback.print_exc()
            try:
                result = await db.execute(select(Document).where(Document.id == document_id))
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = "failed"
                    await db.commit()
                    await invalidate_user_documents(str(user_id))
            except Exception as db_err:
                print(f"Failed to record failure status in db: {db_err}")


@router.get("/")
async def get_documents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Fast path: Redis cache
    cached = await get_cached_user_documents(str(current_user.id))
    if cached is not None:
        return {"documents": cached}

    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id, Document.document_type != "PAST_PAPER")
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    docs_data = [
        {
            "id": str(d.id),
            "filename": d.filename,
            "status": d.status,
            "file_size": d.file_size,
            "mime_type": d.mime_type,
            "document_type": d.document_type,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "year": d.year,
            "subject_id": str(d.subject_id) if d.subject_id else None
        }
        for d in docs
    ]
    await set_cached_user_documents(str(current_user.id), docs_data, ttl=180)
    return {"documents": docs_data}


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

    # Fast path: Redis cache
    cached = await get_cached_doc_topics(document_id)
    if cached is not None:
        return {"topics": cached}

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
    topics_data = [
        {
            "id": str(t.id),
            "name": t.name,
            "summary": t.summary,
            "order_index": t.order_index,
            "document_id": str(t.document_id)
        }
        for t in topics
    ]
    await set_cached_doc_topics(document_id, topics_data, ttl=3600)
    return {"topics": topics_data}


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

    # Fast path: Redis cache
    cached = await get_cached_topic_flashcards(topic_id)
    if cached is not None:
        return cached

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
    payload = {
        "topic": {
            "id": str(topic.id),
            "name": topic.name,
            "summary": topic.summary,
            "document_id": str(topic.document_id)
        },
        "flashcards": [
            {
                "id": str(f.id),
                "question": f.question,
                "answer": f.answer,
                "order_index": f.order_index,
                "topic_id": str(f.topic_id),
                "image_url": f.image_url
            }
            for f in flashcards
        ]
    }
    await set_cached_topic_flashcards(topic_id, payload, ttl=3600)
    return payload



@router.post("/flashcards/{flashcard_id}/flip")
async def flip_flashcard(
    flashcard_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        fc_uuid = uuid.UUID(flashcard_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid flashcard ID")

    result = await db.execute(select(Flashcard).where(Flashcard.id == fc_uuid))
    flashcard = result.scalar_one_or_none()
    if not flashcard:
        raise HTTPException(status_code=404, detail="Flashcard not found")
        
    from sqlalchemy import update
    from services.redis_service import invalidate_user_cache
    await db.execute(
        update(User)
        .where(User.id == current_user.id)
        .values(reputation_credits=User.reputation_credits + 1)
    )
    await db.commit()
    await invalidate_user_cache(str(current_user.id))
    current_user.reputation_credits += 1
    
    return {"message": "Flashcard flipped", "credits": current_user.reputation_credits}


@router.post("/{document_id}/backfill-chunks")
async def backfill_document_chunks(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Backfills DocumentChunk records for an already-processed document.
    Use this for documents uploaded before the chat feature was added.
    Downloads the file from Appwrite, extracts text, and stores chunks.
    """
    from models.chat_models import DocumentChunk

    try:
        doc_uuid = uuid.UUID(document_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(
        select(Document).where(Document.id == doc_uuid, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.status != "completed":
        raise HTTPException(status_code=400, detail="Document is not yet fully processed")

    existing = await db.execute(
        select(DocumentChunk).where(DocumentChunk.document_id == doc_uuid).limit(1)
    )
    if existing.scalar_one_or_none():
        return {"status": "already_exists", "message": "Document chunks already exist"}

    background_tasks.add_task(
        _backfill_chunks_task,
        document_id=doc_uuid,
        user_id=current_user.id
    )
    return {"status": "started", "message": "Chunk backfill started in background"}


async def _backfill_chunks_task(document_id: uuid.UUID, user_id: uuid.UUID):
    """Background task: downloads doc from Appwrite, chunks it, stores DocumentChunk records."""
    from database import AsyncSessionLocal
    from models.chat_models import DocumentChunk
    from services.appwrite_service import download_file_from_appwrite

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(select(Document).where(Document.id == document_id))
            doc = result.scalar_one_or_none()
            if not doc or not doc.appwrite_file_id:
                return

            suffix = os.path.splitext(doc.filename)[1] or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name

            download_file_from_appwrite(doc.appwrite_file_id, temp_path)
            text = extract_text(temp_path, doc.mime_type)
            os.remove(temp_path)

            chunks = chunk_text(text, max_words=500)

            topic_result = await db.execute(
                select(Topic)
                .where(Topic.document_id == document_id, Topic.user_id == user_id)
                .order_by(Topic.order_index.asc())
            )
            topics = topic_result.scalars().all()
            num_chunks = len(chunks)
            num_topics = len(topics)

            for chunk_idx, chunk_content in enumerate(chunks):
                if not chunk_content.strip():
                    continue
                if topics:
                    topic_idx = min(int(chunk_idx / num_chunks * num_topics), num_topics - 1)
                    topic_id = topics[topic_idx].id
                else:
                    topic_id = None

                chunk_rec = DocumentChunk(
                    document_id=document_id,
                    topic_id=topic_id,
                    user_id=user_id,
                    chunk_index=chunk_idx,
                    content=chunk_content.strip()
                )
                db.add(chunk_rec)

            await db.commit()
            print(f"Backfilled {num_chunks} chunks for document {document_id}")

        except Exception as e:
            print(f"Backfill failed for document {document_id}: {e}")
            import traceback
            traceback.print_exc()


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        doc_uuid = uuid.UUID(document_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(
        select(Document).where(Document.id == doc_uuid, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 1. Clean up local file if present
    local_path = os.path.join(UPLOAD_DIR, f"{doc.id}_{doc.filename}")
    if os.path.exists(local_path):
        try:
            os.remove(local_path)
        except Exception as e:
            print(f"Error removing local file {local_path}: {e}")

    # 2. Clean up Appwrite file if present
    if doc.appwrite_file_id:
        try:
            await asyncio.to_thread(delete_file_from_appwrite, doc.appwrite_file_id)
        except Exception as e:
            print(f"Error deleting from Appwrite: {e}")

    # 3. Clean up any extracted images belonging to this document
    try:
        for f in os.listdir(EXTRACTED_IMAGES_DIR):
            if f.startswith(f"{doc.id}_"):
                try:
                    os.remove(os.path.join(EXTRACTED_IMAGES_DIR, f))
                except Exception:
                    pass
    except Exception as img_del_err:
        print(f"Error cleaning extracted images: {img_del_err}")

    # 4. Delete Document (cascade in DB deletes associated topics, flashcards, chunks, chat sessions)
    await db.delete(doc)
    await db.commit()

    # 5. Invalidate caches
    await invalidate_user_documents(str(current_user.id))
    from services.redis_service import redis_client
    if redis_client:
        try:
            await redis_client.delete(f"doc:{document_id}:topics")
        except Exception:
            pass

    return {"message": "Document deleted successfully", "id": document_id}


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        doc_uuid = uuid.UUID(document_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(
        select(Document).where(Document.id == doc_uuid, Document.user_id == current_user.id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Clear any previous partial topics and chunks
    existing_topics = await db.execute(select(Topic).where(Topic.document_id == doc_uuid))
    for t in existing_topics.scalars().all():
        await db.delete(t)

    doc.status = "processing"
    await db.commit()
    await invalidate_user_documents(str(current_user.id))

    local_path = os.path.join(UPLOAD_DIR, f"{doc.id}_{doc.filename}")

    background_tasks.add_task(
        process_document_task,
        document_id=doc.id,
        file_path=local_path,
        mime_type=doc.mime_type,
        user_id=current_user.id
    )

    return {"message": "Reprocessing initiated", "document_id": str(doc.id), "status": "processing"}

