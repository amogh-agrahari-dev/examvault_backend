import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from database import Base

class PastPaper(Base):
    """
    Independent file base for previous year examination papers.
    Strictly isolated from main study documents (documents table).
    """
    __tablename__ = "past_papers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    subject_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("subjects.id"), index=True, nullable=True)

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    exam_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, default="End-Sem")
    extracted_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="processed")

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
    predictions: Mapped[List["PredictedExamPaper"]] = relationship(
        "PredictedExamPaper",
        back_populates="past_paper",
        cascade="all, delete-orphan"
    )


class PredictedExamPaper(Base):
    """
    Predicted examination paper generated independently based solely on the uploaded past paper.
    """
    __tablename__ = "predicted_exam_papers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    subject_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("subjects.id"), index=True, nullable=True)
    past_paper_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("past_papers.id"), index=True, nullable=True)

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    subject_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    target_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_marks: Mapped[int] = mapped_column(Integer, default=70)
    duration_minutes: Mapped[int] = mapped_column(Integer, default=180)
    difficulty: Mapped[str] = mapped_column(String(50), default="medium")

    general_instructions: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    sections: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    marking_scheme: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    summary_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    # Relationship
    past_paper: Mapped[Optional["PastPaper"]] = relationship("PastPaper", back_populates="predictions")
