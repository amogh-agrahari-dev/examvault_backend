import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, Float, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from database import Base

class Quiz(Base):
    __tablename__ = "quizzes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Store selected config as JSON for simplicity
    selected_documents: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True) 
    selected_topics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    question_count: Mapped[int] = mapped_column(Integer, default=10)
    difficulty_distribution: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    question_type_distribution: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    duration: Mapped[int] = mapped_column(Integer, default=30) # in minutes
    scoring_configuration: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    questions: Mapped[List["QuizQuestion"]] = relationship(
        "QuizQuestion",
        back_populates="quiz",
        cascade="all, delete-orphan",
        order_by="QuizQuestion.order_index"
    )
    attempts: Mapped[List["QuizAttempt"]] = relationship(
        "QuizAttempt",
        back_populates="quiz",
        cascade="all, delete-orphan"
    )

class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    quiz_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("quizzes.id"), index=True, nullable=False)
    topic_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("topics.id", ondelete="SET NULL"), nullable=True)
    
    question_type: Mapped[str] = mapped_column(String(50), nullable=False) # mcq, ms, tf, short_answer, coding
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False) # easy, medium, hard
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    
    public_options: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True) # Options or examples
    marks: Mapped[float] = mapped_column(Float, default=1.0)
    
    protected_answer_key: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    grading_rubric: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    protected_coding_evaluation_configuration: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    quiz: Mapped["Quiz"] = relationship("Quiz", back_populates="questions")

class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    quiz_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("quizzes.id"), index=True, nullable=False)
    
    status: Mapped[str] = mapped_column(String(20), default="active") # active, submitted, expired
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expiry_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    submission_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    scoring_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    final_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    evaluation_status: Mapped[str] = mapped_column(String(50), default="pending") # pending, evaluating, completed, failed
    
    quiz: Mapped["Quiz"] = relationship("Quiz", back_populates="attempts")
    answers: Mapped[List["QuizAnswer"]] = relationship(
        "QuizAnswer",
        back_populates="attempt",
        cascade="all, delete-orphan"
    )

class QuizAnswer(Base):
    __tablename__ = "quiz_answers"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attempt_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("quiz_attempts.id"), index=True, nullable=False)
    question_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("quiz_questions.id"), index=True, nullable=False)
    
    student_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    selected_options: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    submitted_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    programming_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    review_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    last_saved_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )
    
    evaluation_status: Mapped[str] = mapped_column(String(50), default="pending")
    awarded_marks: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    evaluation_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    attempt: Mapped["QuizAttempt"] = relationship("QuizAttempt", back_populates="answers")
