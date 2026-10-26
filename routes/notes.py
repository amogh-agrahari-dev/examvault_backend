import os
import uuid
import tempfile
import asyncio
import hashlib
from typing import List, Optional
from datetime import datetime, timezone
import jwt
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Header, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, desc
from sqlalchemy.orm import selectinload

from database import get_db
from config import settings
from routes.auth import get_current_user
from models.auth_models import User
from models.note_models import Note, NoteImage
from models.subject_models import Subject
from models.document_models import Topic, Document
from schemas.note_schemas import (
    NoteCreate,
    NoteUpdate,
    NoteRename,
    NoteListItem,
    NoteDetail,
    NoteImageResponse,
    TopicOption,
    TopicCreateInNote,
)
from services.appwrite_service import upload_file_to_appwrite, storage
from services.redis_service import (
    get_cached_notes,
    set_cached_notes,
    invalidate_user_notes,
    get_cached_user,
    set_cached_user,
)

router = APIRouter(prefix="/api/v1/notes", tags=["notes"])

# Ensure a local directory exists for image caching or safe fallback
LOCAL_IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "note_images")
os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)

MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
ALLOWED_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


async def _get_user_from_token_or_header(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Helper to authenticate requests that may supply a token via header or query parameter."""
    raw_token = None
    if isinstance(authorization, str) and authorization.startswith("Bearer "):
        raw_token = authorization.split(" ", 1)[1]
    elif isinstance(token, str) and token.strip():
        raw_token = token.strip()

    if not raw_token:
        return None

    try:
        try:
            payload = jwt.decode(raw_token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])
        except jwt.ExpiredSignatureError:
            # For persistent image resources embedded in notes, allow valid signatures even if expired
            payload = jwt.decode(
                raw_token,
                settings.JWT_SECRET,
                algorithms=[settings.ALGORITHM],
                options={"verify_exp": False}
            )
        user_id = payload.get("sub")
        if not user_id:
            return None

        # Check Redis session cache first for <1ms authentication
        cached_user = await get_cached_user(str(user_id))
        if cached_user:
            return User(
                id=uuid.UUID(str(cached_user["id"])),
                email=cached_user.get("email"),
                full_name=cached_user.get("full_name") or "",
            )

        result = await db.execute(select(User).where(User.id == uuid.UUID(str(user_id))))
        user = result.scalars().first()
        if user:
            await set_cached_user(str(user.id), {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "created_at": user.created_at.isoformat() if getattr(user, "created_at", None) else None,
            })
        return user
    except Exception as e:
        print(f"Error authenticating user for note image: {e}")
        return None


@router.post("", response_model=NoteDetail, status_code=status.HTTP_201_CREATED)
async def create_note(
    note_in: NoteCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new personal quick note."""
    # Validate optional subject association
    subject_name = None
    if note_in.subject_id:
        subj = await db.get(Subject, note_in.subject_id)
        if not subj or subj.user_id != current_user.id:
            raise HTTPException(status_code=400, detail="Invalid subject ID or subject does not belong to you.")
        subject_name = subj.name

    # Validate optional topic association
    topic_name = None
    if note_in.topic_id:
        top = await db.get(Topic, note_in.topic_id)
        if not top or top.user_id != current_user.id:
            raise HTTPException(status_code=400, detail="Invalid topic ID or topic does not belong to you.")
        topic_name = top.name

    new_note = Note(
        user_id=current_user.id,
        title=note_in.title or "Untitled Note",
        content=note_in.content,
        content_text=note_in.content_text,
        thumbnail_url=note_in.thumbnail_url,
        subject_id=note_in.subject_id,
        topic_id=note_in.topic_id,
    )
    db.add(new_note)
    await db.commit()
    await db.refresh(new_note)

    # Invalidate cached notes for user
    await invalidate_user_notes(str(current_user.id))

    return NoteDetail(
        id=new_note.id,
        title=new_note.title,
        content=new_note.content,
        content_text=new_note.content_text,
        thumbnail_url=new_note.thumbnail_url,
        subject_id=new_note.subject_id,
        subject_name=subject_name,
        topic_id=new_note.topic_id,
        topic_name=topic_name,
        created_at=new_note.created_at,
        updated_at=new_note.updated_at,
    )


@router.get("", response_model=List[NoteListItem])
async def list_notes(
    search: Optional[str] = Query(None),
    subject_id: Optional[uuid.UUID] = Query(None),
    topic_id: Optional[uuid.UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all notes for the authenticated user with optional search and filters."""
    # Fast path: Redis cache
    query_hash = hashlib.md5(f"{search or ''}:{subject_id or ''}:{topic_id or ''}".encode()).hexdigest()
    cached = await get_cached_notes(str(current_user.id), query_hash)
    if cached is not None:
        return cached

    query = (
        select(Note, Subject.name.label("subject_name"), Topic.name.label("topic_name"))
        .outerjoin(Subject, Note.subject_id == Subject.id)
        .outerjoin(Topic, Note.topic_id == Topic.id)
        .where(Note.user_id == current_user.id)
    )

    if subject_id:
        query = query.where(Note.subject_id == subject_id)

    if topic_id:
        query = query.where(Note.topic_id == topic_id)

    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.where(
            or_(
                Note.title.ilike(term),
                Note.content_text.ilike(term)
            )
        )

    query = query.order_by(desc(Note.updated_at))
    result = await db.execute(query)
    rows = result.all()

    items = []
    for note, sub_name, top_name in rows:
        items.append(
            NoteListItem(
                id=note.id,
                title=note.title,
                content_text=note.content_text,
                thumbnail_url=note.thumbnail_url,
                subject_id=note.subject_id,
                subject_name=sub_name,
                topic_id=note.topic_id,
                topic_name=top_name,
                created_at=note.created_at,
                updated_at=note.updated_at,
            )
        )

    # Cache items serialized as dicts with 180s TTL
    cached_data = [item.model_dump() for item in items]
    await set_cached_notes(str(current_user.id), query_hash, cached_data, ttl=180)
    return items



@router.get("/topics", response_model=List[TopicOption])
async def list_available_topics(
    subject_id: Optional[uuid.UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Return topics belonging to the user so they can optionally link a note to a topic."""
    query = (
        select(Topic, Document.subject_id, Document.filename)
        .join(Document, Topic.document_id == Document.id)
        .where(Topic.user_id == current_user.id)
    )
    if subject_id:
        query = query.where(Document.subject_id == subject_id)

    query = query.order_by(Topic.name.asc())
    result = await db.execute(query)
    rows = result.all()

    topics = []
    for topic, doc_subject_id, doc_filename in rows:
        topics.append(
            TopicOption(
                id=topic.id,
                name=topic.name,
                subject_id=doc_subject_id,
                document_name=doc_filename,
            )
        )
    return topics


@router.post("/topics", response_model=TopicOption, status_code=status.HTTP_201_CREATED)
async def create_note_topic(
    topic_in: TopicCreateInNote,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new folder or nested subfolder within a subject for organizing notes."""
    folder_name = topic_in.name.strip()
    if not folder_name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty")

    # Verify subject belongs to user
    subject = await db.get(Subject, topic_in.subject_id)
    if not subject or subject.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Subject not found or does not belong to you")

    # Construct nested folder path if parent_folder provided
    if topic_in.parent_folder and topic_in.parent_folder.strip():
        parent_clean = topic_in.parent_folder.strip().strip("/")
        if not folder_name.startswith(f"{parent_clean} /"):
            full_folder_name = f"{parent_clean} / {folder_name}"
        else:
            full_folder_name = folder_name
    else:
        full_folder_name = folder_name

    # Check for or create notebook document for this subject
    doc_res = await db.execute(
        select(Document).where(
            Document.subject_id == subject.id,
            Document.user_id == current_user.id,
            Document.document_type == "NOTES"
        )
    )
    doc = doc_res.scalars().first()
    if not doc:
        # Check if there is any other document for this subject
        any_doc_res = await db.execute(
            select(Document).where(
                Document.subject_id == subject.id,
                Document.user_id == current_user.id
            )
        )
        doc = any_doc_res.scalars().first()
        if not doc:
            doc = Document(
                user_id=current_user.id,
                subject_id=subject.id,
                filename=f"{subject.name} Notebook",
                document_type="NOTES",
                status="active"
            )
            db.add(doc)
            await db.flush()

    # Check if topic already exists under this document or user with this name
    existing_topic = await db.scalar(
        select(Topic).where(
            Topic.user_id == current_user.id,
            Topic.document_id == doc.id,
            Topic.name == full_folder_name
        )
    )
    if existing_topic:
        return TopicOption(
            id=existing_topic.id,
            name=existing_topic.name,
            subject_id=subject.id,
            document_name=doc.filename,
        )

    new_topic = Topic(
        document_id=doc.id,
        user_id=current_user.id,
        name=full_folder_name,
        summary=f"Folder in {subject.name}",
        order_index=0
    )
    db.add(new_topic)
    await db.commit()
    await db.refresh(new_topic)

    return TopicOption(
        id=new_topic.id,
        name=new_topic.name,
        subject_id=subject.id,
        document_name=doc.filename,
    )



@router.get("/{note_id}", response_model=NoteDetail)
async def get_note(
    note_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve a single note with full document content."""
    query = (
        select(Note, Subject.name.label("subject_name"), Topic.name.label("topic_name"))
        .outerjoin(Subject, Note.subject_id == Subject.id)
        .outerjoin(Topic, Note.topic_id == Topic.id)
        .where(Note.id == note_id, Note.user_id == current_user.id)
    )
    result = await db.execute(query)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Note not found")

    note, sub_name, top_name = row
    return NoteDetail(
        id=note.id,
        title=note.title,
        content=note.content,
        content_text=note.content_text,
        thumbnail_url=note.thumbnail_url,
        subject_id=note.subject_id,
        subject_name=sub_name,
        topic_id=note.topic_id,
        topic_name=top_name,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


@router.put("/{note_id}", response_model=NoteDetail)
async def update_note(
    note_id: uuid.UUID,
    note_in: NoteUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update note content, title, or associations (used by autosave)."""
    note = await db.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Note not found")

    # Validate optional subject association if changed
    if note_in.subject_id is not None:
        if note_in.subject_id:
            subj = await db.get(Subject, note_in.subject_id)
            if not subj or subj.user_id != current_user.id:
                raise HTTPException(status_code=400, detail="Invalid subject")
            note.subject_id = note_in.subject_id
        else:
            note.subject_id = None

    # Validate optional topic association if changed
    if note_in.topic_id is not None:
        if note_in.topic_id:
            top = await db.get(Topic, note_in.topic_id)
            if not top or top.user_id != current_user.id:
                raise HTTPException(status_code=400, detail="Invalid topic")
            note.topic_id = note_in.topic_id
        else:
            note.topic_id = None

    if note_in.title is not None:
        note.title = note_in.title
    if note_in.content is not None:
        note.content = note_in.content
    if note_in.content_text is not None:
        note.content_text = note_in.content_text
    if note_in.thumbnail_url is not None:
        note.thumbnail_url = note_in.thumbnail_url

    note.updated_at = datetime.now(timezone.utc)
    db.add(note)
    await db.commit()
    await db.refresh(note)

    # Invalidate cached notes for user
    await invalidate_user_notes(str(current_user.id))

    # Get subject and topic names
    subject_name = None
    if note.subject_id:
        subj = await db.get(Subject, note.subject_id)
        if subj:
            subject_name = subj.name
    topic_name = None
    if note.topic_id:
        top = await db.get(Topic, note.topic_id)
        if top:
            topic_name = top.name

    return NoteDetail(
        id=note.id,
        title=note.title,
        content=note.content,
        content_text=note.content_text,
        thumbnail_url=note.thumbnail_url,
        subject_id=note.subject_id,
        subject_name=subject_name,
        topic_id=note.topic_id,
        topic_name=topic_name,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


@router.patch("/{note_id}/rename", response_model=NoteDetail)
async def rename_note(
    note_id: uuid.UUID,
    rename_in: NoteRename,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Quickly rename a note."""
    note = await db.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Note not found")

    note.title = rename_in.title.strip()
    note.updated_at = datetime.now(timezone.utc)
    db.add(note)
    await db.commit()
    await db.refresh(note)

    # Invalidate cached notes for user
    await invalidate_user_notes(str(current_user.id))

    subject_name = None
    if note.subject_id:
        subj = await db.get(Subject, note.subject_id)
        if subj:
            subject_name = subj.name
    topic_name = None
    if note.topic_id:
        top = await db.get(Topic, note.topic_id)
        if top:
            topic_name = top.name

    return NoteDetail(
        id=note.id,
        title=note.title,
        content=note.content,
        content_text=note.content_text,
        thumbnail_url=note.thumbnail_url,
        subject_id=note.subject_id,
        subject_name=subject_name,
        topic_id=note.topic_id,
        topic_name=topic_name,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


@router.delete("/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_note(
    note_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a note and cleanly clean up associated image records and storage files."""
    note = await db.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Note not found")

    # Invalidate cached notes for user
    await invalidate_user_notes(str(current_user.id))

    # Fetch associated images
    img_res = await db.execute(select(NoteImage).where(NoteImage.note_id == note_id))
    images = img_res.scalars().all()

    for img in images:
        if img.appwrite_file_id:
            try:
                storage.delete_file(bucket_id=settings.APPWRITE_BUCKET_ID, file_id=img.appwrite_file_id)
            except Exception as e:
                print(f"Warning: Failed to delete image {img.appwrite_file_id} from Appwrite: {e}")
        if img.local_path and os.path.exists(img.local_path):
            try:
                os.remove(img.local_path)
            except Exception:
                pass
        await db.delete(img)

    await db.delete(note)
    await db.commit()
    return None


@router.post("/upload-image", response_model=NoteImageResponse)
async def upload_note_image(
    file: UploadFile = File(...),
    note_id: Optional[uuid.UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload an image for insertion into a note.
    Stores the image in Appwrite (with local file fallback) and returns a persistent URL.
    """
    content_type = file.content_type.lower() if file.content_type else "image/jpeg"
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type ({content_type}). Allowed: PNG, JPG, GIF, WebP."
        )

    # Read image contents
    contents = await file.read()
    file_size = len(contents)
    if file_size > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds maximum allowed size of 5MB.")

    # Determine safe normalized filename (Appwrite bucket requires .jpg instead of .jpeg)
    ext = ALLOWED_MIME_TYPES[content_type]
    safe_name = f"note_img_{uuid.uuid4().hex[:12]}{ext}"

    appwrite_file_id = None
    local_path = None

    # Write to temp file for upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    try:
        # 1. Non-blocking Appwrite upload in worker thread
        try:
            appwrite_file_id = await asyncio.to_thread(upload_file_to_appwrite, temp_path, safe_name)
        except Exception as appwrite_err:
            print(f"Appwrite image upload error: {appwrite_err}. Falling back to local disk storage.")
            # Fallback to local storage
            fallback_path = os.path.join(LOCAL_IMAGE_DIR, safe_name)
            with open(fallback_path, "wb") as f:
                f.write(contents)
            local_path = fallback_path

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


    # If both Appwrite and local fallback failed
    if not appwrite_file_id and not local_path:
        raise HTTPException(status_code=500, detail="Failed to persist image to storage service.")

    # If note_id was specified, verify it belongs to user
    if note_id:
        existing_note = await db.get(Note, note_id)
        if not existing_note or existing_note.user_id != current_user.id:
            note_id = None

    note_image = NoteImage(
        user_id=current_user.id,
        note_id=note_id,
        appwrite_file_id=appwrite_file_id,
        local_path=local_path,
        filename=file.filename or safe_name,
        mime_type=content_type,
        file_size=file_size,
    )
    db.add(note_image)
    await db.commit()
    await db.refresh(note_image)

    image_url = f"/api/v1/notes/images/{note_image.id}"
    return NoteImageResponse(
        id=note_image.id,
        url=image_url,
        filename=note_image.filename,
        file_size=note_image.file_size,
        created_at=note_image.created_at,
    )


@router.get("/images/{image_id}")
async def get_note_image(
    image_id: uuid.UUID,
    token: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Securely serve a stored note image.
    Validates user session (via header or token query parameter) so private note images
    cannot be viewed by unauthorized users.
    """
    # 1. Fetch the image metadata
    note_img = await db.get(NoteImage, image_id)
    if not note_img:
        raise HTTPException(status_code=404, detail="Image not found")

    # 2. Authenticate the caller
    auth_user = await _get_user_from_token_or_header(authorization=authorization, token=token, db=db)
    if not auth_user:
        raise HTTPException(status_code=401, detail="Authentication required to access this note image.")

    # 3. Verify ownership: must match the image uploader
    if note_img.user_id != auth_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this image.")

    image_bytes = None

    # 1. Fast path: check local disk cache first
    local_disk_path = note_img.local_path or os.path.join(LOCAL_IMAGE_DIR, f"{note_img.id}.jpg")
    if os.path.exists(local_disk_path):
        try:
            with open(local_disk_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            print(f"Could not read local image: {e}")

    # 2. Fetch non-blocking from Appwrite if not yet on local disk
    if not image_bytes and note_img.appwrite_file_id:
        try:
            image_bytes = await asyncio.to_thread(
                storage.get_file_view,
                bucket_id=settings.APPWRITE_BUCKET_ID,
                file_id=note_img.appwrite_file_id
            )
            # Write-through to local disk cache for subsequent requests
            if image_bytes:
                try:
                    os.makedirs(os.path.dirname(local_disk_path), exist_ok=True)
                    with open(local_disk_path, "wb") as f:
                        f.write(image_bytes)
                except Exception:
                    pass
        except Exception as e:
            print(f"Could not retrieve image from Appwrite: {e}")

    if not image_bytes:
        raise HTTPException(status_code=404, detail="Image data unavailable")

    return Response(
        content=image_bytes,
        media_type=note_img.mime_type or "image/jpeg",
        headers={
            "Cache-Control": "private, max-age=86400",
            "Content-Disposition": "inline"
        }
    )


class FlowchartAiRequest(BaseModel):
    prompt: str
    model: Optional[str] = None


@router.post("/ai/flowchart")
async def generate_ai_flowchart(
    req: FlowchartAiRequest,
    user: Optional[User] = Depends(_get_user_from_token_or_header),
):
    """
    Leverages Ollama to synthesize a multi-node flowchart graph for revision notes.
    """
    clean_prompt = (req.prompt or "").strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    from services.llm_router import generate_flowchart_graph
    result = await generate_flowchart_graph(clean_prompt, req.model)
    if not result or not result.get("nodes"):
        raise HTTPException(
            status_code=502,
            detail="The AI service was unable to generate a valid flowchart diagram. Please try again."
        )

    return result


class CodeSimulationRequest(BaseModel):
    code: str
    language: Optional[str] = "python"
    stdin: Optional[str] = ""
    model: Optional[str] = None


@router.post("/ai/simulate-code")
async def simulate_code_endpoint(
    req: CodeSimulationRequest,
    user: Optional[User] = Depends(_get_user_from_token_or_header),
):
    """
    Executes and generates an interactive, step-by-step visual simulation of code
    for student notes, including data structures, animated pointers, and variable watch states.
    """
    clean_code = (req.code or "").strip()
    if not clean_code:
        raise HTTPException(status_code=400, detail="Code cannot be empty.")

    from services.code_simulator_service import generate_code_simulation_result
    result = await generate_code_simulation_result(
        code=clean_code,
        language=req.language or "python",
        stdin=req.stdin or "",
        model=req.model,
    )
    return result

