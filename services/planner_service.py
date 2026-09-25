import httpx
import json
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, List, Optional
from config import settings

logger = logging.getLogger(__name__)

async def generate_study_plan_with_gemma(
    days_ahead: int,
    daily_hours: float,
    target_score: int,
    strategy: str,
    documents_context: List[Dict[str, Any]],
    exam_name: str
) -> List[Dict[str, Any]]:
    """
    Leverages Ollama gemma2:9b to synthesize a day-by-day exam preparation schedule
    based on the student's actual uploaded documents, topics, and exam targets.
    """
    days_count = max(1, min(days_ahead, 21)) # Cap between 1 and 21 days
    
    # Build text summary of available documents & topics
    docs_summary_lines = []
    for doc in documents_context:
        topics_str = ", ".join([t.get("name", "") for t in doc.get("topics", []) if t.get("name")])
        if not topics_str:
            topics_str = "General Units & Fundamentals"
        docs_summary_lines.append(f"- Document: '{doc.get('title')}' (ID: {doc.get('id')}) | Topics: {topics_str}")

    docs_context_text = "\n".join(docs_summary_lines) if docs_summary_lines else "- General Engineering Curriculum & Core Subjects"

    prompt = f"""You are an elite academic curriculum strategist and tutor. 
Generate a comprehensive, day-by-day study schedule for a university student preparing for an exam.

EXAM DETAILS:
- Exam Name: {exam_name}
- Days Remaining: {days_count} days
- Daily Study Budget: {daily_hours} hours per day
- Target Score: {target_score}%
- Strategy: {strategy.upper()} (Cramming = high-yield active recall & tests; Balanced = theory + practice; Deep = exhaustive mastery)

COURSE MATERIALS AVAILABLE:
{docs_context_text}

PEDAGOGICAL REQUIREMENTS:
1. Cover {days_count} days (day_index 0 to {days_count - 1}). Day 0 is Today.
2. For each day, allocate 2-3 focused study tasks that sum to approximately {daily_hours} hours.
3. Interleave 4 distinct task types:
   - "reading": Conceptual theory reading and derivation review.
   - "quiz": Active recall diagnostic questions and PYQs.
   - "flashcards": Fast-paced spaced repetition memory sweeps.
   - "mock": Milestone timed mock exams (schedule one mid-way if >= 6 days, and one 1-2 days before the exam).
4. Assign the actual document ID to tasks where relevant.

You MUST respond with valid JSON ONLY. Do not include any explanations or conversational text outside the JSON.
Follow this exact JSON schema:
{{
  "days": [
    {{
      "day_index": 0,
      "tasks": [
        {{
          "title": "Topic or Unit Name — Task Description",
          "task_type": "reading",
          "subject": "Subject Name",
          "document_id": "optional-doc-id-or-null",
          "duration": 1.5,
          "unit": "Unit 1: Fundamentals"
        }}
      ]
    }}
  ]
}}
"""

    ollama_model = settings.OLLAMA_MODEL or "gemma2:9b"
    logger.info(f"Prompting Ollama model '{ollama_model}' for study plan generation ({days_count} days)...")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 4096
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "").strip()

            if response_text:
                try:
                    # Strip any potential markdown wrappers
                    clean_text = response_text
                    if "```json" in clean_text:
                        clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_text:
                        clean_text = clean_text.split("```")[1].split("```")[0].strip()
                    
                    parsed = json.loads(clean_text)
                    days_data = parsed.get("days", [])
                    if isinstance(days_data, list) and len(days_data) > 0:
                        logger.info(f"Successfully generated {len(days_data)} days with Ollama {ollama_model}!")
                        return normalize_generated_days(days_data, days_count, daily_hours, documents_context)
                except Exception as parse_err:
                    logger.warning(f"JSON parsing error from Ollama response: {parse_err}. Falling back to synthesized plan.")

    except Exception as e:
        logger.error(f"Error querying Ollama ({ollama_model}): {e}. Using resilient pedagogical synthesizer.")

    # Resilient fallback synthesis if Ollama is unreachable or model is generating malformed JSON
    return synthesize_fallback_plan(days_count, daily_hours, strategy, documents_context)


def normalize_generated_days(
    days_data: List[Dict[str, Any]],
    target_days: int,
    daily_hours: float,
    documents_context: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Cleans, sorts, and ensures each day adheres strictly to expected structure.
    """
    normalized = []
    fallback_doc_id = documents_context[0]["id"] if documents_context else None
    fallback_subject = documents_context[0]["title"] if documents_context else "Engineering Fundamentals"

    for i in range(target_days):
        matched = next((d for d in days_data if d.get("day_index") == i), None)
        tasks = []
        if matched and isinstance(matched.get("tasks"), list):
            for t_idx, t in enumerate(matched["tasks"]):
                tasks.append({
                    "title": t.get("title", f"Study Session {t_idx + 1}"),
                    "task_type": t.get("task_type", "reading") if t.get("task_type") in ["reading", "quiz", "flashcards", "mock"] else "reading",
                    "subject": t.get("subject", fallback_subject),
                    "document_id": t.get("document_id") if t.get("document_id") and t.get("document_id") != "null" else fallback_doc_id,
                    "duration": float(t.get("duration", max(0.5, daily_hours / 2))),
                    "unit": t.get("unit", f"Unit {t_idx + 1}"),
                    "order_index": t_idx
                })
        else:
            # Generate day tasks if day missing from LLM response
            tasks.append({
                "title": f"Core Concept Synthesis & Problem Solving",
                "task_type": "reading",
                "subject": fallback_subject,
                "document_id": fallback_doc_id,
                "duration": round(daily_hours * 0.6, 1),
                "unit": f"Unit {i + 1}",
                "order_index": 0
            })
            tasks.append({
                "title": f"High-Yield Recall & Diagnostic Quiz",
                "task_type": "quiz",
                "subject": fallback_subject,
                "document_id": fallback_doc_id,
                "duration": round(daily_hours * 0.4, 1),
                "unit": f"Practice Drill",
                "order_index": 1
            })

        normalized.append({
            "day_index": i,
            "tasks": tasks
        })

    return normalized


def synthesize_fallback_plan(
    days_count: int,
    daily_hours: float,
    strategy: str,
    documents_context: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Deterministic pedagogical plan synthesizer using user documents and topics.
    Ensures zero failure even when local LLM is temporarily unavailable.
    """
    plan_days = []
    doc1 = documents_context[0] if documents_context else {"id": None, "title": "Core Syllabus", "topics": []}
    doc2 = documents_context[1] if len(documents_context) > 1 else doc1

    for i in range(days_count):
        is_mock_day = (i == 6 or i == days_count - 1) and days_count >= 5
        curr_doc = doc1 if i % 2 == 0 else doc2
        tasks = []

        if is_mock_day:
            tasks.append({
                "title": f"{'Mid-Term Progress Benchmark' if i == 6 else 'Final Full-Length Simulation'} Mock Exam",
                "task_type": "mock",
                "subject": "Comprehensive Syllabus",
                "document_id": curr_doc.get("id"),
                "duration": round(daily_hours * 0.75, 1),
                "unit": "Timed Exam Blueprint",
                "order_index": 0
            })
            tasks.append({
                "title": "Mock Error Analysis & Retrospective Review",
                "task_type": "reading",
                "subject": curr_doc.get("title", "Review"),
                "document_id": curr_doc.get("id"),
                "duration": round(daily_hours * 0.25, 1),
                "unit": "High-Yield Corrections",
                "order_index": 1
            })
        else:
            tasks.append({
                "title": f"In-Depth Theory: Unit {i + 1} Principles & Derivations",
                "task_type": "reading",
                "subject": curr_doc.get("title", "Engineering"),
                "document_id": curr_doc.get("id"),
                "duration": round(daily_hours * 0.6, 1),
                "unit": f"Unit {i + 1}",
                "order_index": 0
            })
            tasks.append({
                "title": f"Targeted Diagnostic Quiz & Flashcard Sweep",
                "task_type": "quiz" if strategy != "cramming" else "flashcards",
                "subject": curr_doc.get("title", "Engineering"),
                "document_id": curr_doc.get("id"),
                "duration": round(daily_hours * 0.4, 1),
                "unit": f"Unit {i + 1} Active Recall",
                "order_index": 1
            })

        plan_days.append({
            "day_index": i,
            "tasks": tasks
        })

    return plan_days
