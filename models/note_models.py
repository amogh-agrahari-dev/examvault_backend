import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from database import Base


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), default="Untitled Note", nullable=False)
    
    # Structured rich-text document content (JSON representation from TipTap / ProseMirror)
    content: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Plaintext representation for quick searching, previews, and word counts
    content_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Optional thumbnail URL (e.g. first image or diagram in note) for dashboard card display
    thumbnail_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Optional associations with existing ExamVault subjects & topics
    subject_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("subjects.id"), index=True, nullable=True)
    topic_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("topics.id"), index=True, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    user = relationship("User")
    subject = relationship("Subject")
    topic = relationship("Topic")
    images: Mapped[List["NoteImage"]] = relationship("NoteImage", back_populates="note", cascade="all, delete-orphan")


class NoteImage(Base):
    __tablename__ = "note_images"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    note_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("notes.id"), index=True, nullable=True)
    
    appwrite_file_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    local_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    note: Mapped[Optional["Note"]] = relationship("Note", back_populates="images")
    user = relationship("User")
