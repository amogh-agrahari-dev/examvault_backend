import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, Float, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from database import Base


class DocumentChunk(Base):
    """
    Stores pre-processed text chunks from a document, scoped to a topic.
    Populated at document ingestion time so chat-time retrieval is fast.
    chunk_index: order within the document
    topic_id: the topic this chunk belongs to (set during flashcard generation)
    content: the raw text content of this chunk
    """
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    topic_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("topics.id", ondelete="SET NULL"), index=True, nullable=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)

    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


class TopicChatSession(Base):
    """
    Represents a persistent AI chat session grounded in a specific document topic.
    One session per user per topic — upserted on demand.
    """
    __tablename__ = "topic_chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False)
    topic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("topics.id", ondelete="CASCADE"), index=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    messages: Mapped[List["TopicChatMessage"]] = relationship(
        "TopicChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="TopicChatMessage.created_at",
        lazy="selectin"
    )


class TopicChatMessage(Base):
    """
    A single message in a topic-grounded chat session.
    role: 'user' or 'assistant'
    sources: JSON array of source references used for assistant responses
    """
    __tablename__ = "topic_chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("topic_chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # [{document_name, chunk_index, relevance}]

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["TopicChatSession"] = relationship("TopicChatSession", back_populates="messages")
