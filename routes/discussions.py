import uuid
import logging
import json
import asyncio
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
import jwt

from database import get_db, AsyncSessionLocal
from routes.auth import get_current_user
from models.auth_models import User
from models.document_models import Topic, Document
from models.discussion_models import (
    Discussion, 
    DiscussionMember, 
    JoinRequest, 
    DiscussionMessage, 
    DiscussionVisibility, 
    JoinRequestStatus, 
    DiscussionMemberRole,
    ClaimVerification,
    ClaimVerdict
)
from models.chat_models import DocumentChunk
from services.chat_service import (
    retrieve_relevant_chunks,
    build_context_string,
    build_chat_prompt,
    call_ollama_for_chat,
    build_verification_prompt,
    call_ollama_for_verification
)
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/discussions", tags=["discussions"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, discussion_id: str):
        await websocket.accept()
        if discussion_id not in self.active_connections:
            self.active_connections[discussion_id] = set()
        self.active_connections[discussion_id].add(websocket)
        logger.info(f"WS CONNECT: {discussion_id} total_connections={len(self.active_connections[discussion_id])}")

    def disconnect(self, websocket: WebSocket, discussion_id: str):
        if discussion_id in self.active_connections:
            self.active_connections[discussion_id].discard(websocket)
            logger.info(f"WS DISCONNECT: {discussion_id} total_connections={len(self.active_connections[discussion_id])}")
            if not self.active_connections[discussion_id]:
                del self.active_connections[discussion_id]

    async def broadcast_to_room(self, discussion_id: str, message: dict):
        if discussion_id in self.active_connections:
            connections = list(self.active_connections[discussion_id])
            dead_connections = []
            
            async def send(conn: WebSocket):
                try:
                    await conn.send_json(message)
                except Exception:
                    dead_connections.append(conn)
            
            if connections:
                await asyncio.gather(*[send(conn) for conn in connections], return_exceptions=True)
            
            # Prune dead connections immediately to avoid broadcast lag
            if dead_connections and discussion_id in self.active_connections:
                for dead in dead_connections:
                    self.active_connections[discussion_id].discard(dead)
                if not self.active_connections[discussion_id]:
                    del self.active_connections[discussion_id]

manager = ConnectionManager()

class DiscussionCreate(BaseModel):
    title: str
    topic_id: uuid.UUID
    description: Optional[str] = None
    visibility: DiscussionVisibility = DiscussionVisibility.PUBLIC

class DiscussionResponse(BaseModel):
    id: uuid.UUID
    topic_id: uuid.UUID
    creator_id: uuid.UUID
    title: str
    description: Optional[str]
    visibility: str
    created_at: datetime
    member_count: int
    is_member: bool
    role: Optional[str] = None

class MessageCreate(BaseModel):
    content: str
    client_msg_id: Optional[str] = None

class VerifyRequest(BaseModel):
    claim: str
    message_id: Optional[uuid.UUID] = None

class AskAIRequest(BaseModel):
    question: str

class MessageResponse(BaseModel):
    id: uuid.UUID
    discussion_id: uuid.UUID
    user_id: uuid.UUID
    content: str
    created_at: datetime
    user_name: str
    client_msg_id: Optional[str] = None

class JoinRequestResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    user_name: str
    status: str
    created_at: datetime

class MemberResponse(BaseModel):
    user_id: uuid.UUID
    user_name: str
    role: str
    joined_at: datetime

@router.post("", response_model=DiscussionResponse)
async def create_discussion(
    req: DiscussionCreate, 
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    # Verify topic exists
    topic = await db.scalar(select(Topic).where(Topic.id == req.topic_id))
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")

    discussion = Discussion(
        topic_id=req.topic_id,
        creator_id=current_user.id,
        title=req.title,
        description=req.description,
        visibility=req.visibility
    )
    db.add(discussion)
    await db.flush() # flush to get ID

    member = DiscussionMember(
        discussion_id=discussion.id,
        user_id=current_user.id,
        role=DiscussionMemberRole.ADMIN
    )
    db.add(member)
    await db.commit()
    await db.refresh(discussion)
    
    return {
        "id": discussion.id,
        "topic_id": discussion.topic_id,
        "creator_id": discussion.creator_id,
        "title": discussion.title,
        "description": discussion.description,
        "visibility": discussion.visibility,
        "created_at": discussion.created_at,
        "member_count": 1,
        "is_member": True,
        "role": DiscussionMemberRole.ADMIN.value
    }

@router.get("/topic/{topic_id}", response_model=List[DiscussionResponse])
async def list_topic_discussions(
    topic_id: uuid.UUID,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Discussion)
        .where(Discussion.topic_id == topic_id)
        .options(selectinload(Discussion.members))
        .order_by(Discussion.created_at.desc())
    )
    discussions = result.scalars().all()
    
    responses = []
    for d in discussions:
        is_member = any(m.user_id == current_user.id for m in d.members)
        user_role = next((m.role for m in d.members if m.user_id == current_user.id), None)
        responses.append({
            "id": d.id,
            "topic_id": d.topic_id,
            "creator_id": d.creator_id,
            "title": d.title,
            "description": d.description,
            "visibility": d.visibility,
            "created_at": d.created_at,
            "member_count": len(d.members),
            "is_member": is_member,
            "role": user_role
        })
    return responses

@router.get("/all", response_model=List[dict])
async def list_all_discussions(
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Discussion, Topic.name)
        .join(Topic, Discussion.topic_id == Topic.id)
        .options(selectinload(Discussion.members))
        .order_by(Discussion.created_at.desc())
    )
    
    responses = []
    for d, topic_name in result:
        is_member = any(m.user_id == current_user.id for m in d.members)
        user_role = next((m.role for m in d.members if m.user_id == current_user.id), None)
        responses.append({
            "id": str(d.id),
            "topic_id": str(d.topic_id),
            "topic_name": topic_name,
            "creator_id": str(d.creator_id),
            "title": d.title,
            "description": d.description,
            "visibility": d.visibility,
            "created_at": d.created_at.isoformat(),
            "member_count": len(d.members),
            "is_member": is_member,
            "role": user_role
        })
    return responses

@router.post("/{discussion_id}/join")
async def join_discussion(
    discussion_id: uuid.UUID,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    discussion = await db.scalar(select(Discussion).where(Discussion.id == discussion_id))
    if not discussion:
        raise HTTPException(status_code=404, detail="Discussion not found")
        
    if discussion.visibility != DiscussionVisibility.PUBLIC:
        raise HTTPException(status_code=403, detail="Discussion is private, request access instead")
        
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == current_user.id)
    )
    
    if member:
        return {"message": "Already a member"}
        
    new_member = DiscussionMember(
        discussion_id=discussion_id,
        user_id=current_user.id,
        role=DiscussionMemberRole.MEMBER
    )
    db.add(new_member)
    await db.commit()
    return {"message": "Joined successfully"}

@router.post("/{discussion_id}/request")
async def request_join(
    discussion_id: uuid.UUID,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    discussion = await db.scalar(select(Discussion).where(Discussion.id == discussion_id))
    if not discussion:
        raise HTTPException(status_code=404, detail="Discussion not found")
        
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == current_user.id)
    )
    if member:
        raise HTTPException(status_code=400, detail="Already a member")
        
    existing_req = await db.scalar(
        select(JoinRequest)
        .where(
            JoinRequest.discussion_id == discussion_id, 
            JoinRequest.user_id == current_user.id,
            JoinRequest.status == JoinRequestStatus.PENDING
        )
    )
    if existing_req:
        raise HTTPException(status_code=400, detail="Request already pending")
        
    req = JoinRequest(
        discussion_id=discussion_id,
        user_id=current_user.id,
        status=JoinRequestStatus.PENDING
    )
    db.add(req)
    await db.commit()
    return {"message": "Request sent"}

@router.get("/{discussion_id}/requests", response_model=List[JoinRequestResponse])
async def list_requests(
    discussion_id: uuid.UUID,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    # Check admin
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == current_user.id)
    )
    if not member or member.role != DiscussionMemberRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    reqs = await db.execute(
        select(JoinRequest, User)
        .join(User, JoinRequest.user_id == User.id)
        .where(JoinRequest.discussion_id == discussion_id, JoinRequest.status == JoinRequestStatus.PENDING)
    )
    
    results = []
    for req, user in reqs:
        results.append({
            "id": req.id,
            "user_id": req.user_id,
            "user_name": user.full_name,
            "status": req.status,
            "created_at": req.created_at
        })
    return results

@router.post("/{discussion_id}/requests/{request_id}/approve")
async def approve_request(
    discussion_id: uuid.UUID,
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == current_user.id)
    )
    if not member or member.role != DiscussionMemberRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    req = await db.scalar(select(JoinRequest).where(JoinRequest.id == request_id, JoinRequest.discussion_id == discussion_id))
    if not req or req.status != JoinRequestStatus.PENDING:
        raise HTTPException(status_code=404, detail="Pending request not found")
        
    req.status = JoinRequestStatus.APPROVED
    
    new_member = DiscussionMember(
        discussion_id=discussion_id,
        user_id=req.user_id,
        role=DiscussionMemberRole.MEMBER
    )
    db.add(new_member)
    await db.commit()
    return {"message": "Approved"}

@router.post("/{discussion_id}/requests/{request_id}/reject")
async def reject_request(
    discussion_id: uuid.UUID,
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == current_user.id)
    )
    if not member or member.role != DiscussionMemberRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
        
    req = await db.scalar(select(JoinRequest).where(JoinRequest.id == request_id, JoinRequest.discussion_id == discussion_id))
    if not req or req.status != JoinRequestStatus.PENDING:
        raise HTTPException(status_code=404, detail="Pending request not found")
        
    req.status = JoinRequestStatus.REJECTED
    await db.commit()
    return {"message": "Rejected"}

@router.get("/{discussion_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    discussion_id: uuid.UUID,
    response: Response,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = "0"
    response.headers["Pragma"] = "no-cache"
        
    msgs = await db.execute(
        select(DiscussionMessage, User)
        .join(User, DiscussionMessage.user_id == User.id)
        .where(DiscussionMessage.discussion_id == discussion_id)
        .order_by(DiscussionMessage.created_at.asc())
        .limit(200) # Basic limit for now
    )
    
    results = []
    for msg, user in msgs:
        results.append({
            "id": msg.id,
            "discussion_id": msg.discussion_id,
            "user_id": msg.user_id,
            "content": msg.content,
            "created_at": msg.created_at,
            "user_name": user.full_name
        })
    return results

@router.get("/{discussion_id}/members", response_model=List[MemberResponse])
async def list_members(
    discussion_id: uuid.UUID,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    # Any logged in user can see members for now, or restrict to members.
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == current_user.id)
    )
    if not member:
        raise HTTPException(status_code=403, detail="Must be a member to view participants")
        
    result = await db.execute(
        select(DiscussionMember, User)
        .join(User, DiscussionMember.user_id == User.id)
        .where(DiscussionMember.discussion_id == discussion_id)
        .order_by(DiscussionMember.joined_at.asc())
    )
    
    responses = []
    for m, u in result:
        responses.append({
            "user_id": m.user_id,
            "user_name": u.full_name,
            "role": m.role.value if hasattr(m.role, 'value') else str(m.role),
            "joined_at": m.joined_at
        })
    return responses

@router.post("/{discussion_id}/messages", response_model=MessageResponse)
async def send_message(
    discussion_id: uuid.UUID,
    req: MessageCreate,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == current_user.id)
    )
    if not member:
        raise HTTPException(status_code=403, detail="Must be a member to send messages")
        
    msg_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    
    msg = DiscussionMessage(
        id=msg_id,
        discussion_id=discussion_id,
        user_id=current_user.id,
        content=req.content,
        created_at=now,
        updated_at=now
    )
    db.add(msg)
    await db.commit()
    
    msg_dict = {
        "id": str(msg_id),
        "discussion_id": str(discussion_id),
        "user_id": str(current_user.id),
        "content": req.content,
        "created_at": now.isoformat(),
        "user_name": current_user.full_name,
        "client_msg_id": req.client_msg_id
    }
    
    asyncio.create_task(manager.broadcast_to_room(str(discussion_id), {
        "type": "message.created",
        "discussion_id": str(discussion_id),
        "message": msg_dict,
        "client_msg_id": req.client_msg_id
    }))
    
    return msg_dict

async def get_ws_current_user(token: str, db: AsyncSession):
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return await db.scalar(select(User).where(User.id == uuid.UUID(user_id)))
    except Exception:
        return None

@router.websocket("/{discussion_id}/ws")
async def websocket_endpoint(
    websocket: WebSocket, 
    discussion_id: uuid.UUID,
    token: str,
    db: AsyncSession = Depends(get_db)
):
    user = await get_ws_current_user(token, db)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Verify membership
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == user.id)
    )
    if not member:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
        
    disc_id_str = str(discussion_id)
    await manager.connect(websocket, disc_id_str)
    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                data = json.loads(data_text)
            except Exception:
                continue

            msg_type = data.get("type")
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg_type == "message.send":
                raw_content = (data.get("content") or "").strip()
                client_msg_id = data.get("client_msg_id")
                if raw_content:
                    msg_id = uuid.uuid4()
                    now = datetime.now(timezone.utc)
                    async with AsyncSessionLocal() as write_db:
                        new_msg = DiscussionMessage(
                            id=msg_id,
                            discussion_id=discussion_id,
                            user_id=user.id,
                            content=raw_content,
                            created_at=now,
                            updated_at=now
                        )
                        write_db.add(new_msg)
                        await write_db.commit()

                    broadcast_dict = {
                        "id": str(msg_id),
                        "discussion_id": disc_id_str,
                        "user_id": str(user.id),
                        "content": raw_content,
                        "created_at": now.isoformat(),
                        "user_name": user.full_name,
                        "client_msg_id": client_msg_id
                    }
                    asyncio.create_task(manager.broadcast_to_room(disc_id_str, {
                        "type": "message.created",
                        "discussion_id": disc_id_str,
                        "message": broadcast_dict,
                        "client_msg_id": client_msg_id
                    }))
    except WebSocketDisconnect:
        logger.info(f"WS client disconnected: user={user.id} disc={discussion_id}")
    except Exception as e:
        logger.warning(f"WS connection error: {e}")
    finally:
        manager.disconnect(websocket, disc_id_str)


@router.post("/{discussion_id}/verify")
async def verify_claim(
    discussion_id: uuid.UUID,
    req: VerifyRequest,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == current_user.id)
    )
    if not member:
        raise HTTPException(status_code=403, detail="Must be a member to verify claims")

    discussion = await db.scalar(
        select(Discussion).where(Discussion.id == discussion_id)
    )
    if not discussion:
        raise HTTPException(status_code=404, detail="Discussion not found")

    topic = await db.scalar(select(Topic).where(Topic.id == discussion.topic_id))
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
        
    doc = await db.scalar(select(Document).where(Document.id == topic.document_id))
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == topic.document_id,
            DocumentChunk.user_id == topic.user_id
        )
        .order_by(DocumentChunk.chunk_index.asc())
    )
    topic_chunks = chunks_result.scalars().all()

    if not topic_chunks:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "claim": req.claim,
            "explanation": f"The document '{doc.filename}' does not have any readable text chunks available for AI analysis. If this is an older document, please re-upload it.",
            "sources": []
        }

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

    retrieved = retrieve_relevant_chunks(req.claim, chunk_dicts)
    context_str, sources = build_context_string(retrieved)

    prompt = build_verification_prompt(req.claim, context_str)
    
    try:
        result = await call_ollama_for_verification(prompt)
    except Exception as e:
        logger.error(f"Ollama call failed for verification {discussion_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail="The AI service is currently unavailable. Please try again in a moment."
        )

    verdict_str = result.get("verdict", "INSUFFICIENT_EVIDENCE")
    try:
        verdict_enum = ClaimVerdict(verdict_str)
    except ValueError:
        verdict_enum = ClaimVerdict.INSUFFICIENT_EVIDENCE

    verification = ClaimVerification(
        discussion_id=discussion_id,
        topic_id=topic.id,
        message_id=req.message_id,
        user_id=current_user.id,
        claim=req.claim,
        verdict=verdict_enum,
        explanation=result.get("explanation", "No explanation provided.")
    )
    db.add(verification)
    await db.commit()
    await db.refresh(verification)

    resp_data = {
        "id": str(verification.id),
        "verdict": verification.verdict.value if hasattr(verification.verdict, "value") else str(verification.verdict),
        "claim": verification.claim,
        "explanation": verification.explanation,
        "sources": sources,
        "user_id": str(current_user.id),
        "user_name": current_user.full_name,
        "created_at": verification.created_at.isoformat()
    }

    asyncio.create_task(manager.broadcast_to_room(str(discussion_id), {
        "type": "verify.result",
        "discussion_id": str(discussion_id),
        "data": resp_data
    }))

    return resp_data

@router.post("/{discussion_id}/ask-ai")
async def ask_ai(
    discussion_id: uuid.UUID,
    req: AskAIRequest,
    db: AsyncSession = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    member = await db.scalar(
        select(DiscussionMember)
        .where(DiscussionMember.discussion_id == discussion_id, DiscussionMember.user_id == current_user.id)
    )
    if not member:
        raise HTTPException(status_code=403, detail="Must be a member to ask AI")

    discussion = await db.scalar(
        select(Discussion).where(Discussion.id == discussion_id)
    )
    if not discussion:
        raise HTTPException(status_code=404, detail="Discussion not found")

    topic = await db.scalar(select(Topic).where(Topic.id == discussion.topic_id))
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
        
    doc = await db.scalar(select(Document).where(Document.id == topic.document_id))
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks_result = await db.execute(
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == topic.document_id,
            DocumentChunk.user_id == topic.user_id
        )
        .order_by(DocumentChunk.chunk_index.asc())
    )
    topic_chunks = chunks_result.scalars().all()

    if not topic_chunks:
        no_docs_msg = (
            f"The document '{doc.filename}' does not have any readable text chunks available for AI analysis. "
            f"If this is an older document, please re-upload it to enable AI Chat."
        )
        return {
            "answer": no_docs_msg,
            "sources": []
        }

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

    retrieved = retrieve_relevant_chunks(req.question, chunk_dicts)
    context_str, sources = build_context_string(retrieved)

    # We do not have chat history for one-off Ask AI queries in discussions.
    prompt = build_chat_prompt(
        subject_name=doc.filename,
        topic_name=topic.name,
        context_str=context_str,
        conversation_history=[],
        user_question=req.question
    )
    
    try:
        answer = await call_ollama_for_chat(prompt)
    except Exception as e:
        logger.error(f"Ollama call failed for ask_ai {discussion_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail="The AI service is currently unavailable. Please try again in a moment."
        )

    if not answer:
        answer = (
            f"I couldn't generate an answer for \"{topic.name}\" at this moment. "
            "Please try again."
        )

    resp_data = {
        "id": f"ask-{uuid.uuid4()}",
        "question": req.question,
        "answer": answer,
        "sources": sources,
        "user_id": str(current_user.id),
        "user_name": current_user.full_name,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    asyncio.create_task(manager.broadcast_to_room(str(discussion_id), {
        "type": "ask.result",
        "discussion_id": str(discussion_id),
        "data": resp_data
    }))

    return resp_data



