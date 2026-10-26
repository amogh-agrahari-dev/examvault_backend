import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from database import get_db
from routes.auth import get_current_user
from models.auth_models import User
from models.document_models import Topic, Document
from models.quiz_models import Quiz, QuizQuestion, QuizAttempt, QuizAnswer
from schemas.quiz_schemas import (
    QuizCreateRequest, QuizResponse, QuizQuestionResponse, 
    QuizAttemptStartResponse, QuizAnswerUpdate,
    CodeRunRequest, CodeRunResponse, CodeSubmitRequest
)
from services.quiz_service import (
    generate_quiz_questions_llm,
    fetch_context_for_topics,
    evaluate_short_answer_llm,
    evaluate_coding_submission_llm
)
from services.code_sandbox_service import execute_code, evaluate_code_against_tests
from services.redis_service import get_cached_user_quizzes, set_cached_user_quizzes, invalidate_user_quizzes

router = APIRouter(prefix="/api/v1/quiz", tags=["quiz"])

@router.get("/config")
async def get_quiz_config():
    return {
        "difficulties": ["easy", "medium", "hard", "mixed"],
        "question_types": ["mcq", "short_answer", "coding"],
        "max_questions": 50,
        "max_duration": 180
    }

@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(
    request: QuizCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Fetch context
    context = await fetch_context_for_topics(db, request.selected_topics)
    if not context:
        raise HTTPException(status_code=400, detail="Not enough context found for selected topics.")
        
    # Call LLM
    questions_data = await generate_quiz_questions_llm(
        context_text=context,
        question_count=request.question_count,
        difficulty=request.difficulty,
        question_types=request.question_types
    )
    
    if not questions_data:
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions.")
        
    # Save Quiz
    quiz = Quiz(
        user_id=current_user.id,
        title=request.title,
        selected_documents={"ids": request.selected_documents},
        selected_topics={"ids": request.selected_topics},
        question_count=len(questions_data),
        difficulty_distribution={"target": request.difficulty},
        question_type_distribution={"types": request.question_types},
        duration=request.duration,
        scoring_configuration=request.scoring_configuration
    )
    db.add(quiz)
    await db.flush() # get quiz id
    
    # Save Questions
    questions = []
    for idx, q_data in enumerate(questions_data):
        question = QuizQuestion(
            quiz_id=quiz.id,
            question_type=q_data.get("question_type", "mcq"),
            difficulty=q_data.get("difficulty", request.difficulty),
            question_text=q_data.get("question_text", ""),
            public_options=q_data.get("public_options"),
            marks=q_data.get("marks", 1.0),
            protected_answer_key=q_data.get("protected_answer_key"),
            grading_rubric=q_data.get("grading_rubric"),
            order_index=idx
        )
        db.add(question)
        questions.append(question)
        
    await db.commit()
    await invalidate_user_quizzes(str(current_user.id))
    
    # Refresh to return
    result = await db.execute(select(Quiz).where(Quiz.id == quiz.id).options(selectinload(Quiz.questions)))
    saved_quiz = result.scalar_one()
    
    return saved_quiz

@router.post("/{quiz_id}/start", response_model=QuizAttemptStartResponse)
async def start_quiz_attempt(
    quiz_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        q_uuid = uuid.UUID(quiz_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Invalid quiz ID")
        
    result = await db.execute(select(Quiz).where(Quiz.id == q_uuid, Quiz.user_id == current_user.id))
    quiz = result.scalar_one_or_none()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
        
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(minutes=quiz.duration)
    
    attempt = QuizAttempt(
        user_id=current_user.id,
        quiz_id=quiz.id,
        status="active",
        start_timestamp=now,
        expiry_timestamp=expiry
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)
    
    return attempt

@router.put("/attempts/{attempt_id}/answers/{question_id}")
async def save_answer(
    attempt_id: str,
    question_id: str,
    answer_update: QuizAnswerUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        a_uuid = uuid.UUID(attempt_id)
        q_uuid = uuid.UUID(question_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")
        
    # Verify attempt ownership and status
    result = await db.execute(select(QuizAttempt).where(QuizAttempt.id == a_uuid, QuizAttempt.user_id == current_user.id))
    attempt = result.scalar_one_or_none()
    if not attempt or attempt.status != "active":
        raise HTTPException(status_code=400, detail="Invalid or inactive attempt")
        
    if datetime.now(timezone.utc) > attempt.expiry_timestamp:
        attempt.status = "expired"
        await db.commit()
        raise HTTPException(status_code=400, detail="Attempt expired")

    # Save or update answer
    result = await db.execute(select(QuizAnswer).where(QuizAnswer.attempt_id == a_uuid, QuizAnswer.question_id == q_uuid))
    answer = result.scalar_one_or_none()
    
    if not answer:
        answer = QuizAnswer(
            attempt_id=a_uuid,
            question_id=q_uuid
        )
        db.add(answer)
        
    if answer_update.student_answer is not None:
        answer.student_answer = answer_update.student_answer
    if answer_update.selected_options is not None:
        answer.selected_options = answer_update.selected_options
    if answer_update.submitted_code is not None:
        answer.submitted_code = answer_update.submitted_code
    if answer_update.programming_language is not None:
        answer.programming_language = answer_update.programming_language
    if answer_update.review_flag is not None:
        answer.review_flag = answer_update.review_flag
        
    await db.commit()
    return {"status": "success"}

@router.post("/attempts/{attempt_id}/submit")
async def submit_attempt(
    attempt_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        a_uuid = uuid.UUID(attempt_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")
        
    result = await db.execute(select(QuizAttempt).where(QuizAttempt.id == a_uuid, QuizAttempt.user_id == current_user.id).options(selectinload(QuizAttempt.answers)))
    attempt = result.scalar_one_or_none()
    if not attempt or attempt.status != "active":
        raise HTTPException(status_code=400, detail="Invalid or inactive attempt")
        
    attempt.status = "submitted"
    attempt.submission_timestamp = datetime.now(timezone.utc)
    
    # Run immediate evaluation for MCQ (which doesn't need LLM)
    # Get all questions
    q_result = await db.execute(select(QuizQuestion).where(QuizQuestion.quiz_id == attempt.quiz_id))
    questions = {q.id: q for q in q_result.scalars().all()}
    
    total_score = 0
    for ans in attempt.answers:
        q = questions.get(ans.question_id)
        if not q:
            continue

        if q.question_type == 'mcq' and q.protected_answer_key:
            correct = q.protected_answer_key.get("correct_option")
            if ans.selected_options and ans.selected_options.get("selected") == correct:
                ans.awarded_marks = q.marks
                total_score += q.marks
            else:
                ans.awarded_marks = 0.0
            ans.evaluation_status = "completed"

        elif q.question_type == 'coding':
            if ans.submitted_code and ans.submitted_code.strip():
                # Check for structured test cases
                test_cases = []
                if q.protected_coding_evaluation_configuration and isinstance(q.protected_coding_evaluation_configuration, dict):
                    test_cases = q.protected_coding_evaluation_configuration.get("test_cases", [])
                elif q.public_options and isinstance(q.public_options, dict):
                    test_cases = q.public_options.get("test_cases", [])

                if test_cases:
                    eval_res = await evaluate_code_against_tests(
                        code=ans.submitted_code,
                        test_cases=test_cases,
                        language=ans.programming_language or "python"
                    )
                    passed = eval_res.get("passed", 0)
                    total = eval_res.get("total", 1)
                    awarded = (passed / total) * q.marks if total > 0 else 0
                    ans.awarded_marks = round(awarded, 2)
                    ans.evaluation_feedback = f"{passed}/{total} test cases passed."
                else:
                    # Use LLM to accurately grade code semantics, logic, and implementation
                    ref_code = None
                    if q.protected_answer_key and isinstance(q.protected_answer_key, dict):
                        ref_code = q.protected_answer_key.get("correct_code") or q.protected_answer_key.get("code")
                    
                    llm_eval = await evaluate_coding_submission_llm(
                        question_text=q.question_text,
                        student_code=ans.submitted_code,
                        reference_code=ref_code,
                        rubric=q.grading_rubric,
                        max_marks=q.marks
                    )
                    ans.awarded_marks = llm_eval.get("awarded_marks", 0.0)
                    ans.evaluation_feedback = llm_eval.get("feedback", "Code evaluated.")
            else:
                ans.awarded_marks = 0.0
                ans.evaluation_feedback = "No code submitted."

            ans.evaluation_status = "completed"
            total_score += (ans.awarded_marks or 0.0)

        elif q.question_type == 'short_answer':
            if ans.student_answer and ans.student_answer.strip():
                ref_ans = None
                if q.protected_answer_key and isinstance(q.protected_answer_key, dict):
                    ref_ans = q.protected_answer_key.get("correct_answer") or q.protected_answer_key.get("answer")
                
                llm_eval = await evaluate_short_answer_llm(
                    question_text=q.question_text,
                    student_answer=ans.student_answer,
                    reference_answer=ref_ans,
                    rubric=q.grading_rubric,
                    max_marks=q.marks
                )
                ans.awarded_marks = llm_eval.get("awarded_marks", 0.0)
                ans.evaluation_feedback = llm_eval.get("feedback", "Evaluation complete.")
            else:
                ans.awarded_marks = 0.0
                ans.evaluation_feedback = "No answer submitted."

            ans.evaluation_status = "completed"
            total_score += (ans.awarded_marks or 0.0)

        else:
            if ans.awarded_marks is None:
                ans.awarded_marks = 0.0
            ans.evaluation_status = "completed"
            total_score += (ans.awarded_marks or 0.0)
                
    attempt.final_score = total_score
    attempt.evaluation_status = "completed"
    
    await db.commit()
    await invalidate_user_quizzes(str(current_user.id))
    return {"status": "submitted"}

@router.get("", response_model=List[Dict[str, Any]])
async def list_user_quizzes(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Fast path: Redis cache
    cached = await get_cached_user_quizzes(str(current_user.id))
    if cached is not None:
        return cached

    result = await db.execute(
        select(Quiz)
        .where(Quiz.user_id == current_user.id)
        .options(
            selectinload(Quiz.questions),
            selectinload(Quiz.attempts)
        )
        .order_by(Quiz.created_at.desc())
    )
    quizzes = result.scalars().all()
    
    # Collect all doc_ids and topic_ids to batch query names
    doc_ids = set()
    topic_ids = set()
    for q in quizzes:
        if q.selected_documents and isinstance(q.selected_documents.get("ids"), list):
            for did in q.selected_documents["ids"]:
                try:
                    doc_ids.add(uuid.UUID(str(did)))
                except (ValueError, TypeError):
                    pass
        if q.selected_topics and isinstance(q.selected_topics.get("ids"), list):
            for tid in q.selected_topics["ids"]:
                try:
                    topic_ids.add(uuid.UUID(str(tid)))
                except (ValueError, TypeError):
                    pass

    doc_map = {}
    if doc_ids:
        docs_res = await db.execute(select(Document.id, Document.filename).where(Document.id.in_(doc_ids)))
        doc_map = {d.id: d.filename for d in docs_res.all()}

    topic_map = {}
    if topic_ids:
        topics_res = await db.execute(select(Topic.id, Topic.name).where(Topic.id.in_(topic_ids)))
        topic_map = {t.id: t.name for t in topics_res.all()}

    output = []
    for q in quizzes:
        doc_names = []
        if q.selected_documents and isinstance(q.selected_documents.get("ids"), list):
            for did in q.selected_documents["ids"]:
                try:
                    u = uuid.UUID(str(did))
                    if u in doc_map:
                        doc_names.append(doc_map[u])
                except (ValueError, TypeError):
                    pass

        topic_names = []
        if q.selected_topics and isinstance(q.selected_topics.get("ids"), list):
            for tid in q.selected_topics["ids"]:
                try:
                    u = uuid.UUID(str(tid))
                    if u in topic_map:
                        topic_names.append(topic_map[u])
                except (ValueError, TypeError):
                    pass

        total_marks = sum(question.marks for question in q.questions) if q.questions else float(q.question_count)
        
        attempts = sorted(q.attempts, key=lambda a: a.start_timestamp or datetime.min.replace(tzinfo=timezone.utc))
        submitted_attempts = [a for a in attempts if a.status == "submitted" and a.final_score is not None]
        best_score = max((a.final_score for a in submitted_attempts), default=None)
        
        latest_attempt_info = None
        if attempts:
            latest = attempts[-1]
            time_taken = None
            if latest.submission_timestamp and latest.start_timestamp:
                time_taken = (latest.submission_timestamp - latest.start_timestamp).total_seconds()
            latest_attempt_info = {
                "id": str(latest.id),
                "status": latest.status,
                "final_score": latest.final_score,
                "start_timestamp": latest.start_timestamp.isoformat() if latest.start_timestamp else None,
                "submission_timestamp": latest.submission_timestamp.isoformat() if latest.submission_timestamp else None,
                "time_taken": time_taken
            }

        difficulty = "medium"
        if q.difficulty_distribution and isinstance(q.difficulty_distribution, dict):
            difficulty = q.difficulty_distribution.get("target") or "medium"
        elif q.questions and len(q.questions) > 0:
            difficulty = q.questions[0].difficulty

        question_types = ["mcq"]
        if q.question_type_distribution and isinstance(q.question_type_distribution, dict):
            question_types = q.question_type_distribution.get("types") or ["mcq"]
        elif q.questions and len(q.questions) > 0:
            question_types = list({question.question_type for question in q.questions})

        output.append({
            "id": str(q.id),
            "title": q.title,
            "question_count": len(q.questions) if q.questions else q.question_count,
            "duration": q.duration,
            "difficulty": difficulty,
            "question_types": question_types,
            "created_at": q.created_at.isoformat() if q.created_at else None,
            "total_marks": total_marks,
            "document_names": doc_names,
            "topic_names": topic_names,
            "attempts_count": len(attempts),
            "completed_attempts_count": len(submitted_attempts),
            "best_score": best_score,
            "latest_attempt": latest_attempt_info,
            "has_been_attempted": len(attempts) > 0
        })

    await set_cached_user_quizzes(str(current_user.id), output, ttl=180)
    return output

@router.get("/{quiz_id}")
async def get_quiz(
    quiz_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        q_uuid = uuid.UUID(quiz_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid quiz ID")

    result = await db.execute(
        select(Quiz)
        .where(Quiz.id == q_uuid, Quiz.user_id == current_user.id)
        .options(
            selectinload(Quiz.questions),
            selectinload(Quiz.attempts)
        )
    )
    quiz = result.scalar_one_or_none()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    questions_data = [
        {
            "id": str(q.id),
            "topic_id": str(q.topic_id) if q.topic_id else None,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "question_text": q.question_text,
            "public_options": q.public_options,
            "marks": q.marks,
            "order_index": q.order_index
        }
        for q in sorted(quiz.questions, key=lambda x: x.order_index)
    ]

    total_marks = sum(q.marks for q in quiz.questions) if quiz.questions else float(quiz.question_count)
    difficulty = "medium"
    if quiz.difficulty_distribution and isinstance(quiz.difficulty_distribution, dict):
        difficulty = quiz.difficulty_distribution.get("target") or "medium"
    elif quiz.questions and len(quiz.questions) > 0:
        difficulty = quiz.questions[0].difficulty

    question_types = ["mcq"]
    if quiz.question_type_distribution and isinstance(quiz.question_type_distribution, dict):
        question_types = quiz.question_type_distribution.get("types") or ["mcq"]
    elif quiz.questions and len(quiz.questions) > 0:
        question_types = list({q.question_type for q in quiz.questions})

    attempts_data = [
        {
            "id": str(a.id),
            "status": a.status,
            "final_score": a.final_score,
            "start_timestamp": a.start_timestamp.isoformat() if a.start_timestamp else None,
            "submission_timestamp": a.submission_timestamp.isoformat() if a.submission_timestamp else None,
        }
        for a in quiz.attempts
    ]

    return {
        "id": str(quiz.id),
        "title": quiz.title,
        "question_count": len(quiz.questions) if quiz.questions else quiz.question_count,
        "duration": quiz.duration,
        "difficulty": difficulty,
        "question_types": question_types,
        "created_at": quiz.created_at.isoformat() if quiz.created_at else None,
        "total_marks": total_marks,
        "questions": questions_data,
        "attempts": attempts_data
    }

@router.delete("/{quiz_id}")
async def delete_quiz(
    quiz_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        q_uuid = uuid.UUID(quiz_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid quiz ID")

    result = await db.execute(
        select(Quiz).where(Quiz.id == q_uuid, Quiz.user_id == current_user.id)
    )
    quiz = result.scalar_one_or_none()
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    await db.delete(quiz)
    await db.commit()
    await invalidate_user_quizzes(str(current_user.id))
    return {"status": "deleted", "id": str(q_uuid)}

@router.get("/attempts/{attempt_id}/result")
async def get_quiz_result(
    attempt_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        a_uuid = uuid.UUID(attempt_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID")
        
    result = await db.execute(
        select(QuizAttempt)
        .where(QuizAttempt.id == a_uuid, QuizAttempt.user_id == current_user.id)
        .options(selectinload(QuizAttempt.quiz), selectinload(QuizAttempt.answers))
    )
    attempt = result.scalar_one_or_none()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
        
    q_result = await db.execute(
        select(QuizQuestion).where(QuizQuestion.quiz_id == attempt.quiz_id).order_by(QuizQuestion.order_index)
    )
    questions = q_result.scalars().all()
    questions_map = {q.id: q for q in questions}
    total_marks = sum(q.marks for q in questions) if questions else 0.0

    # Sort answers by question order_index
    sorted_answers = sorted(
        attempt.answers,
        key=lambda a: questions_map[a.question_id].order_index if a.question_id in questions_map else 0
    )

    answers_data = []
    for ans in sorted_answers:
        q = questions_map.get(ans.question_id)
        correct_opt = None
        explanation = None
        if q and q.protected_answer_key:
            correct_opt = q.protected_answer_key.get("correct_option")
            explanation = q.protected_answer_key.get("explanation") or q.protected_answer_key.get("correct_answer")

        answers_data.append({
            "question_id": str(ans.question_id),
            "question_text": q.question_text if q else "Question",
            "question_type": q.question_type if q else "mcq",
            "public_options": q.public_options if q else None,
            "correct_option": correct_opt,
            "explanation": explanation or ans.evaluation_feedback,
            "awarded_marks": ans.awarded_marks,
            "marks": q.marks if q else 1.0,
            "student_answer": ans.student_answer or ans.submitted_code,
            "selected_options": ans.selected_options
        })

    result_data = {
        "attempt_id": str(attempt.id),
        "quiz_id": str(attempt.quiz_id),
        "quiz_title": attempt.quiz.title if attempt.quiz else "Quiz",
        "status": attempt.status,
        "score": attempt.final_score,
        "total_marks": total_marks,
        "time_taken": (attempt.submission_timestamp - attempt.start_timestamp).total_seconds() if (attempt.submission_timestamp and attempt.start_timestamp) else None,
        "answers": answers_data
    }
    return result_data

@router.post("/code/run", response_model=CodeRunResponse)
async def run_code(
    request: CodeRunRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes code in an isolated sandbox (Judge0 if configured, or local isolated runner).
    """
    if not request.code or not request.code.strip():
        return {
            "status": "empty",
            "output": "",
            "error": "No code provided to execute.",
            "exit_code": 0,
            "provider": "sandbox"
        }

    result = await execute_code(
        code=request.code,
        language=request.language or "python",
        stdin=request.stdin or ""
    )

    return {
        "status": result.get("status", "success"),
        "output": result.get("output", ""),
        "error": result.get("error", ""),
        "exit_code": result.get("exit_code", 0),
        "provider": result.get("provider", "local_sandbox")
    }

@router.post("/code/submit")
async def submit_code(
    request: CodeSubmitRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submits code for grading against test cases.
    """
    try:
        q_uuid = uuid.UUID(request.question_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid question_id format")

    result = await db.execute(select(QuizQuestion).where(QuizQuestion.id == q_uuid))
    question = result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    test_cases = []
    if question.protected_coding_evaluation_configuration and isinstance(question.protected_coding_evaluation_configuration, dict):
        test_cases = question.protected_coding_evaluation_configuration.get("test_cases", [])
    elif question.public_options and isinstance(question.public_options, dict):
        test_cases = question.public_options.get("test_cases", [])

    if test_cases:
        eval_result = await evaluate_code_against_tests(
            code=request.code,
            test_cases=test_cases,
            language=request.language or "python"
        )
        passed = eval_result["passed"]
        total = eval_result["total"]
        score = round((passed / total * question.marks), 2) if total > 0 else 0.0
        return {
            "score": score,
            "total_marks": question.marks,
            "passed": passed,
            "total": total,
            "test_results": eval_result["details"],
            "status": "passed" if eval_result["all_passed"] else "failed"
        }
    else:
        exec_res = await execute_code(
            code=request.code,
            language=request.language or "python"
        )
        has_error = bool(exec_res.get("error")) or exec_res.get("exit_code") != 0
        score = question.marks if not has_error else 0.0
        return {
            "score": score,
            "total_marks": question.marks,
            "output": exec_res.get("output"),
            "error": exec_res.get("error"),
            "status": "passed" if not has_error else "failed"
        }
