import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from database import Base

class SummaryJob(Base):
    __tablename__ = "summary_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    
    selected_documents: Mapped[dict] = mapped_column(JSON, nullable=False) 
    selected_topics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True) 
    
    status: Mapped[str] = mapped_column(String(50), default="pending") 
    progress: Mapped[int] = mapped_column(Integer, default=0) 
    current_stage: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Store discovered topics here before generation
    discovered_topics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    job_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("summary_jobs.id", ondelete="SET NULL"), nullable=True)
    
    topic_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Structured summary content based on prompt
    content: Mapped[dict] = mapped_column(JSON, nullable=False) 
    
    generation_model: Mapped[str] = mapped_column(String(100), default="gemma2:9b")
    source_documents: Mapped[dict] = mapped_column(JSON, nullable=False) 
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )
