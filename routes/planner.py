import uuid
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from database import get_db
from routes.auth import get_current_user
from models.auth_models import User
from models.document_models import Document, Topic
from models.planner_models import StudyPlan, StudyTask
from services.planner_service import generate_study_plan_with_gemma

router = APIRouter(prefix="/api/v1/planner", tags=["planner"])

# Pydantic Schemas
class GeneratePlanRequest(BaseModel):
    exam_name: str = Field(default="SPPU Examination 2026")
    exam_date: str = Field(description="Target exam date in YYYY-MM-DD")
    daily_hours: float = Field(default=3.0, ge=0.5, le=16.0)
    target_score: int = Field(default=90, ge=40, le=100)
    strategy: str = Field(default="balanced", description="cramming, balanced, deep")
    document_ids: Optional[List[str]] = Field(default=None)

class AddTaskRequest(BaseModel):
    day_index: int = Field(default=0)
    title: str
    task_type: str = Field(default="reading")
    subject: str = Field(default="General")
    document_id: Optional[str] = None
    duration: float = Field(default=1.0)
    unit: Optional[str] = None


def format_plan_response(plan: StudyPlan) -> Dict[str, Any]:
    today = date.today()
    days_left = max(0, (plan.exam_date - today).days)
    
    # Group tasks by day_index
    days_map = {}
    for task in plan.tasks:
        d_idx = task.day_index
        if d_idx not in days_map:
            task_date = plan.created_at.date() + timedelta(days=d_idx)
            is_today = task_date == today
            is_tomorrow = task_date == (today + timedelta(days=1))
            is_mock_day = any(t.task_type == "mock" for t in plan.tasks if t.day_index == d_idx)
            days_map[d_idx] = {
                "dayIndex": d_idx,
                "dateStr": task_date.strftime("%a, %b %-d"),
                "isToday": is_today,
                "isTomorrow": is_tomorrow,
                "isMockDay": is_mock_day,
                "tasks": []
            }
        
        days_map[d_idx]["tasks"].append({
            "id": str(task.id),
            "title": task.title,
            "type": task.task_type,
            "subject": task.subject,
            "docId": str(task.document_id) if task.document_id else None,
            "duration": task.duration,
            "unit": task.unit,
            "completed": task.completed
        })

    sorted_days = [days_map[k] for k in sorted(days_map.keys())]

    total_tasks = len(plan.tasks)
    completed_tasks = sum(1 for t in plan.tasks if t.completed)
    completion_percent = round((completed_tasks / total_tasks * 100)) if total_tasks > 0 else 0
    readiness_score = min(96, round(52 + (completion_percent * 0.44)))

    return {
        "id": str(plan.id),
        "examName": plan.exam_name,
        "examDate": plan.exam_date.isoformat(),
        "dailyHours": plan.daily_hours,
        "targetScore": plan.target_score,
        "strategy": plan.strategy,
        "daysRemaining": days_left,
        "totalTasks": total_tasks,
        "completedTasks": completed_tasks,
        "completionPercent": completion_percent,
        "readinessScore": readiness_score,
        "days": sorted_days
    }


@router.get("/current")
async def get_current_plan(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves the currently active study plan and associated tasks for the user from Neon DB.
    """
    stmt = (
        select(StudyPlan)
        .options(selectinload(StudyPlan.tasks))
        .where(StudyPlan.user_id == current_user.id, StudyPlan.is_active == True)
        .order_by(StudyPlan.created_at.desc())
    )
    result = await db.execute(stmt)
    plan = result.scalars().first()

    if not plan:
        return {"plan": None}

    return {"plan": format_plan_response(plan)}


@router.post("/generate")
async def generate_plan(
    req: GeneratePlanRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Uses Ollama (gemma2:9b) to generate a personalized study plan from user's course documents in Neon.
    """
    try:
        parsed_exam_date = datetime.strptime(req.exam_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid exam_date format. Must be YYYY-MM-DD")

    today = date.today()
    days_ahead = max(1, (parsed_exam_date - today).days)

    # 1. Fetch user's documents and detected topics from Neon
    doc_query = (
        select(Document)
        .options(selectinload(Document.topics))
        .where(Document.user_id == current_user.id)
    )
    doc_res = await db.execute(doc_query)
    user_docs = doc_res.scalars().all()

    documents_context = []
    for d in user_docs:
        if req.document_ids and str(d.id) not in req.document_ids:
            continue
        documents_context.append({
            "id": str(d.id),
            "title": d.filename.replace(".pdf", "").replace(".docx", ""),
            "topics": [{"name": t.name, "summary": t.summary} for t in d.topics]
        })

    # If user has no uploaded documents yet, provide standard high-yield academic syllabus context
    if not documents_context:
        documents_context = [
            {"id": None, "title": "Engineering Mathematics", "topics": [{"name": "Linear Algebra"}, {"name": "Differential Equations"}, {"name": "Laplace Transforms"}]},
            {"id": None, "title": "Computer Networks", "topics": [{"name": "OSI Model"}, {"name": "TCP/IP Protocol"}, {"name": "Routing Algorithms"}]}
        ]

    # 2. Generate syllabus breakdown with Ollama gemma2:9b
    generated_days = await generate_study_plan_with_gemma(
        days_ahead=days_ahead,
        daily_hours=req.daily_hours,
        target_score=req.target_score,
        strategy=req.strategy,
        documents_context=documents_context,
        exam_name=req.exam_name
    )

    # 3. Archive any existing active plans in Neon DB
    existing_stmt = select(StudyPlan).where(StudyPlan.user_id == current_user.id, StudyPlan.is_active == True)
    existing_res = await db.execute(existing_stmt)
    for old_plan in existing_res.scalars().all():
        old_plan.is_active = False

    # 4. Create new StudyPlan record in Neon DB
    new_plan = StudyPlan(
        user_id=current_user.id,
        exam_name=req.exam_name,
        exam_date=parsed_exam_date,
        daily_hours=req.daily_hours,
        target_score=req.target_score,
        strategy=req.strategy,
        is_active=True
    )
    db.add(new_plan)
    await db.flush()

    # 5. Populate StudyTask records
    tasks_to_add = []
    for day in generated_days:
        d_idx = day.get("day_index", 0)
        scheduled_date = today + timedelta(days=d_idx)
        for t_idx, t in enumerate(day.get("tasks", [])):
            doc_id_val = None
            if t.get("document_id"):
                try:
                    doc_id_val = uuid.UUID(t.get("document_id"))
                except (ValueError, TypeError):
                    doc_id_val = None

            task_obj = StudyTask(
                plan_id=new_plan.id,
                user_id=current_user.id,
                day_index=d_idx,
                scheduled_date=scheduled_date,
                title=t.get("title", f"Study Task {t_idx + 1}"),
                task_type=t.get("task_type", "reading"),
                subject=t.get("subject", "General"),
                document_id=doc_id_val,
                duration=float(t.get("duration", 1.0)),
                unit=t.get("unit"),
                order_index=t_idx,
                completed=False
            )
            tasks_to_add.append(task_obj)

    db.add_all(tasks_to_add)
    await db.commit()

    # Re-fetch with loaded tasks
    stmt = (
        select(StudyPlan)
        .options(selectinload(StudyPlan.tasks))
        .where(StudyPlan.id == new_plan.id)
    )
    res = await db.execute(stmt)
    saved_plan = res.scalar_one()

    return {"plan": format_plan_response(saved_plan), "message": "Study plan generated successfully with Gemma 2: 9B"}


@router.patch("/tasks/{task_id}/toggle")
async def toggle_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Toggles a task's completion status in Neon DB and updates Vault credits.
    """
    try:
        t_uuid = uuid.UUID(task_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Study task not found")

    stmt = select(StudyTask).where(StudyTask.id == t_uuid, StudyTask.user_id == current_user.id)
    res = await db.execute(stmt)
    task = res.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Study task not found")

    task.completed = not task.completed
    task.completed_at = datetime.now(timezone.utc) if task.completed else None

    # Award / deduct 10 Vault credits
    credits_delta = 10 if task.completed else -10
    current_user.reputation_credits = max(0, current_user.reputation_credits + credits_delta)

    await db.commit()
    await db.refresh(task)

    return {
        "task_id": str(task.id),
        "completed": task.completed,
        "credits": current_user.reputation_credits,
        "message": "+10 Vault Credits awarded!" if task.completed else "Task marked as incomplete"
    }


@router.post("/rebalance")
async def rebalance_plan(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Redistributes incomplete past tasks into future days without student penalty.
    """
    stmt = (
        select(StudyPlan)
        .options(selectinload(StudyPlan.tasks))
        .where(StudyPlan.user_id == current_user.id, StudyPlan.is_active == True)
        .order_by(StudyPlan.created_at.desc())
    )
    result = await db.execute(stmt)
    plan = result.scalars().first()

    if not plan:
        raise HTTPException(status_code=404, detail="No active study plan found")

    today = date.today()
    max_day_index = max((t.day_index for t in plan.tasks), default=0)

    # Shift incomplete tasks from day 0 or overdue
    rebalanced_count = 0
    next_slot = 1
    for task in plan.tasks:
        if not task.completed and (task.day_index == 0 or task.scheduled_date < today):
            task.day_index = min(max_day_index, next_slot)
            task.scheduled_date = today + timedelta(days=task.day_index)
            if not task.title.startswith("[Catch-Up]"):
                task.title = f"[Catch-Up] {task.title}"
            next_slot = (next_slot + 1) % max(2, max_day_index + 1)
            rebalanced_count += 1

    await db.commit()
    await db.refresh(plan)

    return {
        "plan": format_plan_response(plan),
        "rebalanced_tasks": rebalanced_count,
        "message": f"Successfully rebalanced {rebalanced_count} tasks across upcoming days."
    }


@router.post("/tasks")
async def add_custom_task(
    req: AddTaskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Adds a custom task or AI remedial drill to the user's active plan.
    """
    stmt = (
        select(StudyPlan)
        .where(StudyPlan.user_id == current_user.id, StudyPlan.is_active == True)
        .order_by(StudyPlan.created_at.desc())
    )
    result = await db.execute(stmt)
    plan = result.scalars().first()

    if not plan:
        raise HTTPException(status_code=404, detail="No active study plan found")

    doc_id_val = None
    if req.document_id:
        try:
            doc_id_val = uuid.UUID(req.document_id)
        except (ValueError, TypeError):
            doc_id_val = None

    task = StudyTask(
        plan_id=plan.id,
        user_id=current_user.id,
        day_index=req.day_index,
        scheduled_date=date.today() + timedelta(days=req.day_index),
        title=req.title,
        task_type=req.task_type,
        subject=req.subject,
        document_id=doc_id_val,
        duration=req.duration,
        unit=req.unit,
        completed=False
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)

    return {
        "task": {
            "id": str(task.id),
            "title": task.title,
            "type": task.task_type,
            "subject": task.subject,
            "docId": str(task.document_id) if task.document_id else None,
            "duration": task.duration,
            "unit": task.unit,
            "completed": task.completed
        },
        "message": "Study task added successfully."
    }
