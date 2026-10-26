import uuid
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class NoteCreate(BaseModel):
    title: Optional[str] = "Untitled Note"
    content: Optional[Any] = None
    content_text: Optional[str] = None
    thumbnail_url: Optional[str] = None
    subject_id: Optional[uuid.UUID] = None
    topic_id: Optional[uuid.UUID] = None


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[Any] = None
    content_text: Optional[str] = None
    thumbnail_url: Optional[str] = None
    subject_id: Optional[uuid.UUID] = None
    topic_id: Optional[uuid.UUID] = None
    client_timestamp: Optional[str] = None  # To prevent out-of-order race conditions


class NoteRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class NoteListItem(BaseModel):
    id: uuid.UUID
    title: str
    content_text: Optional[str] = None
    thumbnail_url: Optional[str] = None
    subject_id: Optional[uuid.UUID] = None
    subject_name: Optional[str] = None
    topic_id: Optional[uuid.UUID] = None
    topic_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NoteDetail(BaseModel):
    id: uuid.UUID
    title: str
    content: Optional[Any] = None
    content_text: Optional[str] = None
    thumbnail_url: Optional[str] = None
    subject_id: Optional[uuid.UUID] = None
    subject_name: Optional[str] = None
    topic_id: Optional[uuid.UUID] = None
    topic_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class NoteImageResponse(BaseModel):
    id: uuid.UUID
    url: str
    filename: str
    file_size: int
    created_at: datetime

    class Config:
        from_attributes = True


class TopicOption(BaseModel):
    id: uuid.UUID
    name: str
    subject_id: Optional[uuid.UUID] = None
    document_name: Optional[str] = None

    class Config:
        from_attributes = True


class TopicCreateInNote(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    subject_id: uuid.UUID
    parent_folder: Optional[str] = None

