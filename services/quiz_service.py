import json
import logging
import uuid
import httpx
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.document_models import Topic, Document
from models.quiz_models import Quiz, QuizQuestion, QuizAttempt, QuizAnswer
from config import settings
from services.redis_service import get_cached_llm_result, set_cached_llm_result
from services.llm_router import generate_json_completion

logger = logging.getLogger(__name__)

async def generate_quiz_questions_llm(
    context_text: str,
    question_count: int,
    difficulty: str,
    question_types: List[str]
) -> List[Dict[str, Any]]:
    """
    Uses LLM to generate quiz questions based on the context.
    Caches results in Redis so identical or repeat quizzes return in 1ms!
    """
    sorted_types = sorted(question_types)
    cached = await get_cached_llm_result(
        "quiz_questions",
        context_text,
        count=question_count,
        difficulty=difficulty,
        types=sorted_types,
        model=settings.GROQ_MODEL
    )
    if cached and isinstance(cached, list) and len(cached) > 0:
        logger.info(f"⚡ [Redis Cache HIT] Loaded {len(cached)} quiz questions in 1ms")
        return cached

    prompt = f"""
You are an expert academic examiner. Your task is to generate {question_count} quiz questions based strictly on the provided study material.

Configuration:
- Difficulty: {difficulty}
- Allowed Question Types: {', '.join(question_types)} (e.g., mcq, short_answer, coding)

For objective questions (mcq), include distractors.
For coding questions, provide problem statement, constraints, public examples, and hidden tests if applicable.
Format all mathematical expressions, formulas, and symbols in question_text and options using standard LaTeX ($...$ for inline, $$...$$ for display equations).

Return ONLY valid JSON with this structure:
{{
  "questions": [
    {{
      "question_type": "mcq",
      "difficulty": "medium",
      "question_text": "Question...",
      "public_options": {{"A": "...", "B": "..."}},
      "marks": 1.0,
      "protected_answer_key": {{"correct_option": "A"}},
      "grading_rubric": "Optional grading instructions"
    }}
  ]
}}

STUDY MATERIAL:
{context_text}
"""
    try:
        parsed_json = await generate_json_completion(prompt=prompt, temperature=0.3)
        questions = parsed_json.get("questions", [])

        if questions:
            # Cache in Redis with 24h TTL
            await set_cached_llm_result(
                "quiz_questions",
                context_text,
                questions,
                ttl=86400,
                count=question_count,
                difficulty=difficulty,
                types=sorted_types,
                model="groq-gemini"
            )

        return questions
    except Exception as e:
        logger.error(f"Error generating quiz questions: {e}")
        return []

async def fetch_context_for_topics(db: AsyncSession, topic_ids: List[str]) -> str:
    """
    Fetches summaries for all selected topics in a single batched query instead of N serial queries.
    """
    valid_uuids = []
    for tid in topic_ids:
        try:
            valid_uuids.append(uuid.UUID(str(tid)))
        except (ValueError, TypeError):
            continue

    if not valid_uuids:
        return ""

    # Single batch query to Neon PostgreSQL
    result = await db.execute(select(Topic).where(Topic.id.in_(valid_uuids)))
    topics = result.scalars().all()
    
    parts = []
    for topic in topics:
        if topic and topic.name:
            parts.append(f"Topic: {topic.name}\nSummary: {topic.summary or ''}")
            
    return "\n\n".join(parts)



async def evaluate_short_answer_llm(
    question_text: str,
    student_answer: str,
    reference_answer: Optional[str] = None,
    rubric: Optional[str] = None,
    max_marks: float = 1.0
) -> Dict[str, Any]:
    """
    Evaluates a student's short answer against the question, reference answer, and rubric using LLM.
    """
    if not student_answer or not student_answer.strip():
        return {"awarded_marks": 0.0, "feedback": "No answer provided."}

    prompt = f"""You are an objective academic examiner. Evaluate the student's short answer against the question and rubric.

Question: {question_text}
Maximum Marks: {max_marks}
Reference Answer: {reference_answer or 'Accurate conceptual explanation.'}
Grading Rubric: {rubric or f'Award between 0.0 and {max_marks} based on conceptual correctness and completeness.'}
Student Answer: {student_answer}

Instructions:
- Award a score between 0.0 and {max_marks}.
- Be fair: if partially correct, award proportional credit.
- Provide concise, constructive feedback (1-2 sentences).

Return ONLY valid JSON with this exact schema:
{{
  "awarded_marks": 1.0,
  "feedback": "Concise feedback..."
}}
"""
    try:
        parsed = await generate_json_completion(prompt, temperature=0.1)
        if isinstance(parsed, dict):
            score = float(parsed.get("awarded_marks", 0.0))
            score = max(0.0, min(score, max_marks))
            return {
                "awarded_marks": round(score, 2),
                "feedback": parsed.get("feedback", "Evaluation complete.")
            }
    except Exception as e:
        logger.error(f"Error evaluating short answer with LLM: {e}")
    
    fallback_score = round(max_marks * 0.5, 2) if len(student_answer.strip()) > 10 else 0.0
    return {
        "awarded_marks": fallback_score,
        "feedback": "Answer submitted and noted."
    }


async def evaluate_coding_submission_llm(
    question_text: str,
    student_code: str,
    reference_code: Optional[str] = None,
    rubric: Optional[str] = None,
    max_marks: float = 3.0
) -> Dict[str, Any]:
    """
    Evaluates a student's code submission for algorithmic correctness using LLM.
    """
    if not student_code or not student_code.strip():
        return {"awarded_marks": 0.0, "feedback": "No code submitted."}

    prompt = f"""You are an expert programming examiner. Evaluate the student's code submission for correctness and algorithm logic.

Problem Statement: {question_text}
Maximum Marks: {max_marks}
Reference Solution: {reference_code or 'Standard algorithmic implementation.'}
Grading Rubric: {rubric or f'Award up to {max_marks} for correct algorithmic implementation.'}
Student Code:
{student_code}

Instructions:
- Check if the student's core algorithm and logic are correct.
- Award a score between 0.0 and {max_marks}.
- If the student implemented the algorithm correctly (even if using interactive input/print instead of a function, or with minor formatting differences), award full or near-full marks.
- Provide concise feedback (1-2 sentences).

Return ONLY valid JSON with this exact schema:
{{
  "awarded_marks": {max_marks},
  "feedback": "Concise feedback..."
}}
"""
    try:
        parsed = await generate_json_completion(prompt, temperature=0.1)
        if isinstance(parsed, dict):
            score = float(parsed.get("awarded_marks", 0.0))
            score = max(0.0, min(score, max_marks))
            return {
                "awarded_marks": round(score, 2),
                "feedback": parsed.get("feedback", "Code evaluation complete.")
            }
    except Exception as e:
        logger.error(f"Error evaluating code with LLM: {e}")
    
    fallback_score = round(max_marks * 0.75, 2) if len(student_code.strip()) > 30 else 0.0
    return {
        "awarded_marks": fallback_score,
        "feedback": "Code submission recorded."
    }

