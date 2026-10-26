import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from database import Base
from typing import List, Optional

class Subject(Base):
    __tablename__ = "subjects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    documents: Mapped[List["Document"]] = relationship("Document", back_populates="subject", cascade="all, delete-orphan")
    concepts: Mapped[List["ExtractedConcept"]] = relationship("ExtractedConcept", back_populates="subject", cascade="all, delete-orphan")
    past_questions: Mapped[List["PastPaperQuestion"]] = relationship("PastPaperQuestion", back_populates="subject", cascade="all, delete-orphan")

class ExtractedConcept(Base):
    __tablename__ = "extracted_concepts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("subjects.id"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    frequency: Mapped[int] = mapped_column(Integer, default=1)
    recency_score: Mapped[float] = mapped_column(Float, default=0.0)
    importance_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    subject: Mapped["Subject"] = relationship("Subject", back_populates="concepts")
    questions: Mapped[List["PastPaperQuestion"]] = relationship("PastPaperQuestion", back_populates="concept", cascade="all, delete-orphan")

class PastPaperQuestion(Base):
    __tablename__ = "past_paper_questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("subjects.id"), index=True, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id"), index=True, nullable=False)
    concept_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("extracted_concepts.id"), index=True, nullable=True)
    
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    marks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    question_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    difficulty: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    subject: Mapped["Subject"] = relationship("Subject", back_populates="past_questions")
    document: Mapped["Document"] = relationship("Document", back_populates="past_questions")
    concept: Mapped["ExtractedConcept"] = relationship("ExtractedConcept", back_populates="questions")

