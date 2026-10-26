"""
Topic-Grounded AI Chat Routes
==============================
Provides endpoints for managing topic-scoped chat sessions and sending messages.
Every request is authenticated and scoped to the current user — users can never
access another user's sessions, topics, or document chunks.

Endpoints:
    GET  /api/v1/chat/topics              — List all user's documents with topics
    POST /api/v1/chat/sessions            — Get or create a chat session for a topic
    GET  /api/v1/chat/sessions/{id}       — Get session with message history
    POST /api/v1/chat/sessions/{id}/ask   — Ask a question (RAG pipeline)
    DELETE /api/v1/chat/sessions/{id}     — Clear chat history for a session
"""

import uuid
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from database import get_db
from routes.auth import get_current_user
from models.auth_models import User
from models.document_models import Document, Topic
from models.chat_models import DocumentChunk, TopicChatSession, TopicChatMessage
from services.chat_service import (
    retrieve_relevant_chunks,
    build_context_string,
    build_chat_prompt,
    call_ollama_for_chat
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ──────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    topic_id: str


class AskRequest(BaseModel):
    question: str


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _serialize_message(msg: TopicChatMessage) -> dict:
    return {
        "id": str(msg.id),
        "role": msg.role,
        "content": msg.content,
        "sources": msg.sources,
        "created_at": msg.created_at.isoformat(),
    }


def _serialize_session(session: TopicChatSession, topic: Topic, document: Document) -> dict:
    return {
        "id": str(session.id),
        "topic_id": str(session.topic_id),
        "topic_name": topic.name,
        "document_id": str(session.document_id),
        "document_name": document.filename,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


async def _verify_topic_ownership(topic_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> Topic:
    """Ensures the topic belongs to the current user. Raises 404 if not found or not owned."""
    result = await db.execute(
        select(Topic).where(Topic.id == topic_id, Topic.user_id == user_id)
    )
    topic = result.scalar_one_or_none()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    return topic


async def _verify_session_ownership(session_id: uuid.UUID, user_id: uuid.UUID, db: AsyncSession) -> TopicChatSession:
    """Ensures the session belongs to the current user."""
    result = await db.execute(
        select(TopicChatSession)
        .where(TopicChatSession.id == session_id, TopicChatSession.user_id == user_id)
        .options(selectinload(TopicChatSession.messages))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session


# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────

@router.get("/topics")
async def get_chat_topics(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns all of the user's completed documents with their topics.
    Used by the frontend to populate the topic selector.
    """
    result = await db.execute(
        select(Document)
        .where(Document.user_id == current_user.id, Document.status == "completed")
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()

    output = []
    for doc in docs:
        topic_result = await db.execute(
            select(Topic)
            .where(Topic.document_id == doc.id, Topic.user_id == current_user.id)
            .order_by(Topic.order_index.asc())
        )
        topics = topic_result.scalars().all()
        output.append({
            "document_id": str(doc.id),
            "document_name": doc.filename,
            "topics": [
                {"id": str(t.id), "name": t.name, "summary": t.summary, "order_index": t.order_index}
                for t in topics
            ]
        })

    return {"documents": output}


@router.post("/sessions")
async def get_or_create_session(
    req: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get an existing chat session for a topic, or create one if it doesn't exist.
    One session per user per topic.
    """
    try:
        topic_uuid = uuid.UUID(req.topic_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid topic_id format")

    # Verify topic belongs to user
    topic = await _verify_topic_ownership(topic_uuid, current_user.id, db)

    # Fetch the parent document (also verify ownership)
    doc_result = await db.execute(
        select(Document).where(Document.id == topic.document_id, Document.user_id == current_user.id)
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Associated document not found")

    # Check for existing session
    session_result = await db.execute(
        select(TopicChatSession)
        .where(
            TopicChatSession.user_id == current_user.id,
            TopicChatSession.topic_id == topic_uuid
        )
        .options(selectinload(TopicChatSession.messages))
    )
    session = session_result.scalar_one_or_none()

    if not session:
        session = TopicChatSession(
            user_id=current_user.id,
            document_id=topic.document_id,
            topic_id=topic_uuid
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        session.messages = []
        messages_data = []
    else:
        messages_data = [_serialize_message(m) for m in (session.messages or [])]

    return {
        "session": _serialize_session(session, topic, doc),
        "messages": messages_data
    }


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve a session and its full message history."""
    try:
        session_uuid = uuid.UUID(session_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = await _verify_session_ownership(session_uuid, current_user.id, db)

    topic_result = await db.execute(select(Topic).where(Topic.id == session.topic_id))
    topic = topic_result.scalar_one_or_none()
    doc_result = await db.execute(select(Document).where(Document.id == session.document_id))
    doc = doc_result.scalar_one_or_none()

    if not topic or not doc:
        raise HTTPException(status_code=404, detail="Session references deleted topic or document")

    return {
        "session": _serialize_session(session, topic, doc),
        "messages": [_serialize_message(m) for m in (session.messages or [])]
    }


@router.post("/sessions/{session_id}/ask")
async def ask_question(
    session_id: str,
    req: AskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    RAG pipeline:
      1. Validate auth & session ownership
      2. Load topic context
      3. Retrieve relevant document chunks (scoped to this topic + user)
      4. Build grounded prompt with conversation history
      5. Call Ollama LLM
      6. Persist user + assistant messages
      7. Return answer + sources
    """
    try:
        session_uuid = uuid.UUID(session_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Question too long (max 2000 characters)")

    # 1. Verify ownership and load session + messages
    session = await _verify_session_ownership(session_uuid, current_user.id, db)

    # 2. Load topic and document (both must belong to user)
    topic_result = await db.execute(
        select(Topic).where(Topic.id == session.topic_id, Topic.user_id == current_user.id)
    )
    topic = topic_result.scalar_one_or_none()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found or access denied")

    doc_result = await db.execute(
        select(Document).where(Document.id == session.document_id, Document.user_id == current_user.id)
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found or access denied")

    # 3. Retrieve ALL document chunks for this document owned by the user.
    #    This allows the RAG pipeline to search the entire uploaded file
    #    rather than artificially restricting to proportionally mapped chunks.
    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == session.document_id,
            DocumentChunk.user_id == current_user.id
        )
        .order_by(DocumentChunk.chunk_index.asc())
    )
    topic_chunks = chunks_result.scalars().all()

    # 4. Handle "no documents uploaded" case
    if not topic_chunks:
        no_docs_msg = (
            f"The document '{doc.filename}' does not have any readable text chunks available for AI analysis. "
            f"If this is an older document, please re-upload it to enable AI Chat and Claim Verification."
        )
        # Persist user message
        user_msg = TopicChatMessage(
            session_id=session.id,
            role="user",
            content=question,
            sources=None
        )
        db.add(user_msg)

        # Persist assistant response
        asst_msg = TopicChatMessage(
            session_id=session.id,
            role="assistant",
            content=no_docs_msg,
            sources=[]
        )
        db.add(asst_msg)
        await db.commit()

        return {
            "answer": no_docs_msg,
            "sources": [],
            "session_id": str(session.id)
        }

    # 5. BM25 retrieval — find most relevant chunks
    chunk_dicts = [
        {
            "id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "document_name": doc.filename,
            "topic_id": str(chunk.topic_id) if chunk.topic_id else None,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content
        }
        for chunk in topic_chunks
    ]

    retrieved = retrieve_relevant_chunks(question, chunk_dicts)

    # 6. Build context string
    context_str, sources = build_context_string(retrieved)

    # 7. Build conversation history for the prompt
    recent_messages = session.messages[-10:] if session.messages else []  # last 10 messages
    history = [{"role": m.role, "content": m.content} for m in recent_messages]

    # 8. Build the grounded prompt
    prompt = build_chat_prompt(
        subject_name=doc.filename,
        topic_name=topic.name,
        context_str=context_str,
        conversation_history=history,
        user_question=question
    )

    # 9. Call Ollama
    try:
        answer = await call_ollama_for_chat(prompt)
    except Exception as e:
        logger.error(f"Ollama call failed for session {session_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail="The AI service is currently unavailable. Please try again in a moment."
        )

    if not answer:
        answer = (
            f"I couldn't generate an answer for \"{topic.name}\" at this moment. "
            "Please try again."
        )

    # 10. Persist user message + assistant message
    user_msg = TopicChatMessage(
        session_id=session.id,
        role="user",
        content=question,
        sources=None
    )
    db.add(user_msg)

    asst_msg = TopicChatMessage(
        session_id=session.id,
        role="assistant",
        content=answer,
        sources=sources if sources else []
    )
    db.add(asst_msg)
    await db.commit()

    return {
        "answer": answer,
        "sources": sources,
        "session_id": str(session.id)
    }


@router.delete("/sessions/{session_id}")
async def clear_chat_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Clears all messages in a chat session without deleting the session itself.
    The session stays associated with the topic.
    """
    try:
        session_uuid = uuid.UUID(session_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = await _verify_session_ownership(session_uuid, current_user.id, db)

    # Delete all messages
    for msg in (session.messages or []):
        await db.delete(msg)
    await db.commit()

    return {"status": "cleared", "session_id": str(session.id)}
