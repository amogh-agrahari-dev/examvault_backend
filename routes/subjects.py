import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from database import get_db
from models.auth_models import User
from models.subject_models import Subject
from models.document_models import Document
from models.past_paper_models import PastPaper
from routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/subjects", tags=["subjects"])

class SubjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class SubjectResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    created_at: str
    document_count: Optional[int] = 0
    past_paper_count: Optional[int] = 0

    class Config:
        from_attributes = True

@router.post("", response_model=SubjectResponse)
async def create_subject(
    subject: SubjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Check if subject with same name exists for this user
    existing = await db.scalar(
        select(Subject).where(Subject.user_id == current_user.id, Subject.name == subject.name)
    )
    if existing:
        raise HTTPException(status_code=400, detail="Subject with this name already exists")
    
    new_subject = Subject(
        user_id=current_user.id,
        name=subject.name,
        description=subject.description
    )
    db.add(new_subject)
    await db.commit()
    await db.refresh(new_subject)
    
    return {
        "id": new_subject.id,
        "name": new_subject.name,
        "description": new_subject.description,
        "created_at": new_subject.created_at.isoformat(),
        "document_count": 0,
        "past_paper_count": 0
    }

@router.get("", response_model=List[SubjectResponse])
async def list_subjects(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Get all subjects
    result = await db.execute(select(Subject).where(Subject.user_id == current_user.id))
    subjects = result.scalars().all()
    
    response = []
    from sqlalchemy import func
    for sub in subjects:
        # Count documents
        doc_count = await db.scalar(
            select(func.count(Document.id))
            .where(Document.subject_id == sub.id, Document.document_type == "NOTES")
        )
        pp_count = await db.scalar(
            select(func.count(PastPaper.id))
            .where(PastPaper.subject_id == sub.id)
        )
        
        response.append({
            "id": sub.id,
            "name": sub.name,
            "description": sub.description,
            "created_at": sub.created_at.isoformat(),
            "document_count": doc_count or 0,
            "past_paper_count": pp_count or 0
        })
        
    return response

@router.delete("/{subject_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_subject(
    subject_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    subject = await db.get(Subject, subject_id)
    if not subject:
        raise HTTPException(status_code=404, detail="Subject not found")
    if subject.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    await db.delete(subject)
    await db.commit()
    return None
