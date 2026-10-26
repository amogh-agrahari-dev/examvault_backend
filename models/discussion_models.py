import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import String, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from database import Base
import enum

class DiscussionVisibility(str, enum.Enum):
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"

class JoinRequestStatus(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class DiscussionMemberRole(str, enum.Enum):
    ADMIN = "ADMIN"
    MEMBER = "MEMBER"

class Discussion(Base):
    __tablename__ = "discussions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("topics.id", ondelete="CASCADE"), index=True, nullable=False)
    creator_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    visibility: Mapped[DiscussionVisibility] = mapped_column(String(50), default=DiscussionVisibility.PUBLIC)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    members: Mapped[List["DiscussionMember"]] = relationship("DiscussionMember", back_populates="discussion", cascade="all, delete-orphan")
    messages: Mapped[List["DiscussionMessage"]] = relationship("DiscussionMessage", back_populates="discussion", cascade="all, delete-orphan")
    join_requests: Mapped[List["JoinRequest"]] = relationship("JoinRequest", back_populates="discussion", cascade="all, delete-orphan")


class DiscussionMember(Base):
    __tablename__ = "discussion_members"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    discussion_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("discussions.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    role: Mapped[DiscussionMemberRole] = mapped_column(String(50), default=DiscussionMemberRole.MEMBER)

    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    discussion: Mapped["Discussion"] = relationship("Discussion", back_populates="members")
    # user relationship can be implicit or queried


class JoinRequest(Base):
    __tablename__ = "join_requests"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    discussion_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("discussions.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    status: Mapped[JoinRequestStatus] = mapped_column(String(50), default=JoinRequestStatus.PENDING)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    discussion: Mapped["Discussion"] = relationship("Discussion", back_populates="join_requests")


class DiscussionMessage(Base):
    __tablename__ = "discussion_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    discussion_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("discussions.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    discussion: Mapped["Discussion"] = relationship("Discussion", back_populates="messages")


class ClaimVerdict(str, enum.Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

class ClaimVerification(Base):
    __tablename__ = "claim_verifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    discussion_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("discussions.id", ondelete="CASCADE"), index=True, nullable=False)
    topic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("topics.id", ondelete="CASCADE"), index=True, nullable=False)
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("discussion_messages.id", ondelete="SET NULL"), nullable=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[ClaimVerdict] = mapped_column(String(50), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    discussion: Mapped["Discussion"] = relationship("Discussion")
    message: Mapped[Optional["DiscussionMessage"]] = relationship("DiscussionMessage")
