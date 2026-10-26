import uuid
import logging
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, or_

from database import get_db
from routes.auth import get_current_user
from models.auth_models import User
from models.document_models import Document, Topic
from models.quiz_models import QuizAttempt
from models.discussion_models import Discussion, DiscussionMember, DiscussionMessage

from models.note_models import Note

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/users", tags=["users"])


# ──── Response models ────────────────────────────────────────────────────

class PublicProfileResponse(BaseModel):
    id: str
    full_name: str
    role: str
    reputation_credits: int
    joined_at: str
    documents_count: int
    topics_count: int
    quizzes_taken: int
    avg_quiz_score: Optional[float]
    discussions_joined: int
    messages_sent: int

class UserSearchResult(BaseModel):
    id: str
    full_name: str
    role: str
    reputation_credits: int

class DiscussionSearchResult(BaseModel):
    id: str
    title: str
    topic_name: Optional[str]
    member_count: int
    visibility: str

class DocumentSearchResult(BaseModel):
    id: str
    filename: str
    document_type: str

class NoteSearchResult(BaseModel):
    id: str
    title: str
    content_preview: Optional[str]

class GlobalSearchResponse(BaseModel):
    users: List[UserSearchResult]
    discussions: List[DiscussionSearchResult]
    documents: List[DocumentSearchResult]
    notes: List[NoteSearchResult]


class LeaderboardUserResponse(BaseModel):
    rank: int
    id: str
    full_name: str
    role: str
    reputation_credits: int
    joined_at: str
    quizzes_taken: int
    is_current_user: bool


# ──── Leaderboard ────────────────────────────────────────────────────────

@router.get("/leaderboard", response_model=List[LeaderboardUserResponse])
async def get_credits_leaderboard(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Returns top users ranked by reputation credits."""
    query = (
        select(User)
        .order_by(User.reputation_credits.desc(), User.created_at.asc())
        .limit(limit)
    )
    result = await db.execute(query)
    users = result.scalars().all()

    leaderboard = []
    for rank, u in enumerate(users, start=1):
        # Count quizzes taken
        q_count = await db.scalar(
            select(func.count(QuizAttempt.id)).where(
                QuizAttempt.user_id == u.id,
                QuizAttempt.status == "submitted"
            )
        ) or 0

        leaderboard.append(
            LeaderboardUserResponse(
                rank=rank,
                id=str(u.id),
                full_name=u.full_name,
                role=u.role,
                reputation_credits=u.reputation_credits,
                joined_at=u.created_at.strftime("%b %Y") if u.created_at else "Member",
                quizzes_taken=q_count,
                is_current_user=u.id == current_user.id
            )
        )
    return leaderboard


# ──── Public Profile ─────────────────────────────────────────────────────

@router.get("/{user_id}/profile", response_model=PublicProfileResponse)
async def get_user_profile(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a public profile for any user."""
    target_user = await db.scalar(select(User).where(User.id == user_id))
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Document count
    docs_count = await db.scalar(
        select(func.count(Document.id)).where(Document.user_id == user_id)
    ) or 0

    # Topics count
    topics_count = await db.scalar(
        select(func.count(Topic.id)).where(Topic.user_id == user_id)
    ) or 0

    # Quiz attempts
    quiz_stats = await db.execute(
        select(
            func.count(QuizAttempt.id),
            func.avg(QuizAttempt.final_score)
        ).where(
            QuizAttempt.user_id == user_id,
            QuizAttempt.status == "submitted"
        )
    )
    quiz_row = quiz_stats.first()
    quizzes_taken = quiz_row[0] or 0
    avg_score = round(float(quiz_row[1]), 1) if quiz_row[1] is not None else None

    # Discussions joined
    discussions_joined = await db.scalar(
        select(func.count(DiscussionMember.id)).where(DiscussionMember.user_id == user_id)
    ) or 0

    # Messages sent
    messages_sent = await db.scalar(
        select(func.count(DiscussionMessage.id)).where(DiscussionMessage.user_id == user_id)
    ) or 0

    return {
        "id": str(target_user.id),
        "full_name": target_user.full_name,
        "role": target_user.role,
        "reputation_credits": target_user.reputation_credits,
        "joined_at": target_user.created_at.isoformat() if target_user.created_at else datetime.now(timezone.utc).isoformat(),
        "documents_count": docs_count,
        "topics_count": topics_count,
        "quizzes_taken": quizzes_taken,
        "avg_quiz_score": avg_score,
        "discussions_joined": discussions_joined,
        "messages_sent": messages_sent,
    }


# ──── Global Search ──────────────────────────────────────────────────────

@router.get("/search/global", response_model=GlobalSearchResponse)
async def global_search(
    q: str = Query(..., min_length=1, max_length=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search users and discussions globally."""
    query_lower = f"%{q.lower()}%"

    # Search users by name or email
    user_results = await db.execute(
        select(User)
        .where(
            or_(
                func.lower(User.full_name).like(query_lower),
                func.lower(User.email).like(query_lower)
            )
        )
        .limit(10)
    )
    users = [
        {
            "id": str(u.id),
            "full_name": u.full_name,
            "role": u.role,
            "reputation_credits": u.reputation_credits,
        }
        for u in user_results.scalars().all()
    ]

    # Search discussions by title
    disc_query = (
        select(Discussion, Topic.name.label("topic_name"))
        .outerjoin(Topic, Discussion.topic_id == Topic.id)
        .where(func.lower(Discussion.title).like(query_lower))
        .limit(10)
    )
    disc_results = await db.execute(disc_query)

    discussions = []
    for row in disc_results.all():
        disc = row[0]
        topic_name = row[1]
        member_count = await db.scalar(
            select(func.count(DiscussionMember.id)).where(DiscussionMember.discussion_id == disc.id)
        ) or 0
        discussions.append({
            "id": str(disc.id),
            "title": disc.title,
            "topic_name": topic_name,
            "member_count": member_count,
            "visibility": disc.visibility.value if hasattr(disc.visibility, 'value') else str(disc.visibility),
        })

    # Search documents
    doc_results = await db.execute(
        select(Document)
        .where(
            Document.user_id == current_user.id,
            func.lower(Document.filename).like(query_lower)
        )
        .limit(8)
    )
    documents = [
        {
            "id": str(d.id),
            "filename": d.filename,
            "document_type": d.document_type or "NOTES"
        }
        for d in doc_results.scalars().all()
    ]

    # Search notes
    note_results = await db.execute(
        select(Note)
        .where(
            Note.user_id == current_user.id,
            or_(
                func.lower(Note.title).like(query_lower),
                func.lower(Note.content_text).like(query_lower)
            )
        )
        .limit(8)
    )
    notes = [
        {
            "id": str(n.id),
            "title": n.title,
            "content_preview": (n.content_text[:120] + "...") if n.content_text and len(n.content_text) > 120 else (n.content_text or "")
        }
        for n in note_results.scalars().all()
    ]

    return {
        "users": users,
        "discussions": discussions,
        "documents": documents,
        "notes": notes,
    }
