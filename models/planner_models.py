import uuid
from datetime import datetime, timezone, date
from typing import List, Optional
from sqlalchemy import String, Integer, DateTime, Date, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from database import Base

class StudyPlan(Base):
    __tablename__ = "study_plans"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    exam_name: Mapped[str] = mapped_column(String(255), nullable=False)
    exam_date: Mapped[date] = mapped_column(Date, nullable=False)
    daily_hours: Mapped[float] = mapped_column(Float, default=3.0)
    target_score: Mapped[int] = mapped_column(Integer, default=90)
    strategy: Mapped[str] = mapped_column(String(50), default="balanced")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    tasks: Mapped[List["StudyTask"]] = relationship(
        "StudyTask",
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="StudyTask.day_index, StudyTask.order_index"
    )


class StudyTask(Base):
    __tablename__ = "study_tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("study_plans.id"), index=True, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    day_index: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 = today, 1 = tomorrow, etc.
    scheduled_date: Mapped[date] = mapped_column(Date, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), default="reading")  # reading, quiz, flashcards, mock, custom
    subject: Mapped[str] = mapped_column(String(255), default="General")
    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    duration: Mapped[float] = mapped_column(Float, default=1.0)
    unit: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    plan: Mapped["StudyPlan"] = relationship("StudyPlan", back_populates="tasks")
