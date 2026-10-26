import os
import io
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Awaitable
from PIL import Image

from config import settings
from services.redis_service import get_cached_llm_result, set_cached_llm_result

logger = logging.getLogger(__name__)

# Initialize Groq client
_groq_client = None
try:
    from groq import AsyncGroq, RateLimitError, APIError
    if settings.GROQ_API_KEY:
        _groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
except Exception as e:
    logger.warning(f"Could not initialize AsyncGroq client: {e}")

# Initialize Gemini
_gemini_configured = False
try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import google.generativeai as genai
    gemini_key = settings.GEMINI_API_KEY.strip() if settings.GEMINI_API_KEY else ""
    if gemini_key and gemini_key != "your_gemini_api_key":
        genai.configure(api_key=gemini_key)
        _gemini_configured = True
        logger.info("Google Gemini configured successfully.")
    else:
        logger.info("GEMINI_API_KEY not configured or using placeholder.")
except Exception as e:
    logger.warning(f"Could not configure Google Generative AI: {e}")


PRIMARY_GROQ_MODELS = [
    settings.GROQ_MODEL,
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3.8-27b"
]
PRIMARY_GEMINI_MODEL = settings.GEMINI_MODEL or "gemini-1.5-flash"


async def generate_completion(
    prompt: str,
    is_multimodal: bool = False,
    image_bytes: Optional[bytes] = None,
    system_prompt: Optional[str] = None,
    json_format: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    model_override: Optional[str] = None
) -> str:
    """
    Intelligent dual-provider LLM router:
    1. If multimodal (image provided or is_multimodal=True), routes to Gemini 1.5 Flash.
    2. If text-only, routes to Groq (ultra-fast inference).
    3. If Groq hits rate limits or errors, automatically falls back to Gemini 1.5 Flash.
    4. If cloud providers fail/offline, attempts local Ollama fallback if available.
    """
    # ── 1. Multimodal Routing: Google Gemini 1.5 Flash ──────────────────────
    if is_multimodal or image_bytes is not None:
        return await _call_gemini_multimodal(
            prompt=prompt,
            image_bytes=image_bytes,
            system_prompt=system_prompt,
            json_format=json_format,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_override or PRIMARY_GEMINI_MODEL
        )

    # ── 2. Text-only Routing: Groq with automatic Gemini Fallback ────────────
    if _groq_client:
        try:
            return await _call_groq(
                prompt=prompt,
                system_prompt=system_prompt,
                json_format=json_format,
                temperature=temperature,
                max_tokens=max_tokens,
                model_name=model_override
            )
        except Exception as groq_err:
            logger.warning(f"Groq API call encountered error ({groq_err}). Executing fallback to Gemini...")

    # Fallback to Gemini for text completion
    try:
        return await _call_gemini_text(
            prompt=prompt,
            system_prompt=system_prompt,
            json_format=json_format,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=PRIMARY_GEMINI_MODEL
        )
    except Exception as gemini_err:
        logger.warning(f"Gemini fallback failed ({gemini_err}). Checking local Ollama fallback...")

    # Final dev fallback: local Ollama
    return await _call_ollama_fallback(prompt=prompt, json_format=json_format, temperature=temperature)


async def _call_groq(
    prompt: str,
    system_prompt: Optional[str] = None,
    json_format: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    model_name: Optional[str] = None
) -> str:
    """Executes text completion on Groq with model failover."""
    models_to_try = [model_name] if model_name else PRIMARY_GROQ_MODELS
    last_err = None

    for model in models_to_try:
        if not model:
            continue
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if json_format:
                kwargs["response_format"] = {"type": "json_object"}

            response = await _groq_client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            last_err = e
            # 1. Recover completed JSON if Groq returned json_validate_failed with failed_generation
            try:
                if hasattr(e, "response") and e.response is not None:
                    err_json = e.response.json()
                    failed_gen = err_json.get("error", {}).get("failed_generation")
                    if failed_gen and isinstance(failed_gen, str) and failed_gen.strip():
                        logger.info(f"Successfully recovered generated JSON payload from Groq on model {model}.")
                        return failed_gen.strip()
            except Exception:
                pass

            err_msg = str(e).lower()
            # If rate limit (429), raise immediately to trigger Gemini fallback
            if "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg:
                logger.warning(f"Groq Rate Limit (429) hit on model {model}. Triggering Gemini fallback.")
                raise e
            # If model not found (404), try next model in candidate list
            if "404" in err_msg or "does not exist" in err_msg:
                continue
            raise e

    if last_err:
        raise last_err
    raise RuntimeError("No Groq model could be executed.")


async def _call_gemini_text(
    prompt: str,
    system_prompt: Optional[str] = None,
    json_format: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    model_name: str = "gemini-1.5-flash"
) -> str:
    """Executes text completion on Google Gemini."""
    if not _gemini_configured:
        raise RuntimeError("Google Gemini is not configured with a valid GEMINI_API_KEY in .env.")

    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }
    if json_format:
        generation_config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
        generation_config=generation_config
    )
    response = await model.generate_content_async(prompt)
    return (response.text or "").strip()


async def _call_gemini_multimodal(
    prompt: str,
    image_bytes: Optional[bytes],
    system_prompt: Optional[str] = None,
    json_format: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    model_name: str = "gemini-1.5-flash"
) -> str:
    """Executes vision / multimodal analysis on Google Gemini."""
    if not _gemini_configured:
        raise RuntimeError("Google Gemini is not configured with a valid GEMINI_API_KEY in .env.")

    if not image_bytes:
        return await _call_gemini_text(
            prompt=prompt,
            system_prompt=system_prompt,
            json_format=json_format,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_name
        )

    pil_img = Image.open(io.BytesIO(image_bytes))
    if pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")

    generation_config: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens
    }
    if json_format:
        generation_config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
        generation_config=generation_config
    )
    response = await model.generate_content_async([prompt, pil_img])
    return (response.text or "").strip()


async def _call_ollama_fallback(prompt: str, json_format: bool = False, temperature: float = 0.2) -> str:
    """Local dev fallback if cloud providers are unreachable or keys missing."""
    import httpx
    async with httpx.AsyncClient(timeout=45.0) as client:
        payload: Dict[str, Any] = {
            "model": settings.OLLAMA_MODEL or "llama3.2:latest",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
        if json_format:
            payload["format"] = "json"
        res = await client.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload)
        res.raise_for_status()
        return res.json().get("response", "").strip()


async def generate_json_completion(
    prompt: str,
    is_multimodal: bool = False,
    image_bytes: Optional[bytes] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    model_override: Optional[str] = None
) -> Dict[str, Any]:
    """Generates completion and parses JSON output with robust markdown stripping."""
    raw = await generate_completion(
        prompt=prompt,
        is_multimodal=is_multimodal,
        image_bytes=image_bytes,
        system_prompt=system_prompt,
        json_format=True,
        temperature=temperature,
        max_tokens=max_tokens,
        model_override=model_override
    )
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    if "```json" in raw:
        try:
            cleaned = raw.split("```json")[1].split("```")[0].strip()
            return json.loads(cleaned)
        except Exception:
            pass

    if "```" in raw:
        try:
            cleaned = raw.split("```")[1].split("```")[0].strip()
            return json.loads(cleaned)
        except Exception:
            pass

    s_idx = raw.find("{")
    e_idx = raw.rfind("}")
    if s_idx != -1 and e_idx != -1 and e_idx > s_idx:
        try:
            return json.loads(raw[s_idx:e_idx + 1])
        except Exception:
            pass

    raise ValueError(f"Could not parse valid JSON from LLM response: {raw[:200]}...")


# ── High-Level Service Functions Migrated from Ollama Service ────────────────

async def _process_flashcard_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    semaphore: asyncio.Semaphore,
    progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
) -> Optional[Dict[str, Any]]:
    """Processes a study text chunk through LLM Router with Redis caching and concurrency throttling."""
    cached = await get_cached_llm_result("flashcards_chunk", chunk, model="groq-gemini")
    if cached:
        logger.info(f"⚡ [Redis Cache HIT] Flashcards for chunk {chunk_index + 1}/{total_chunks}")
        if progress_callback:
            try:
                await progress_callback(chunk_index + 1, total_chunks)
            except Exception:
                pass
        return cached

    prompt = f"""You are an expert academic tutor. Convert this study chunk into high-yield revision flashcards.
1. Be Exhaustive: Do not omit key definitions, formulas, or concepts.
2. Group into structured logical topics.
3. Use LaTeX formatting ($...$ or $$...$$) for all math and formulas.
4. If the chunk mentions `[Embedded Figure / Diagram: /api/v1/documents/images/...]`, create a flashcard testing that diagram and set `"image_url"` to that exact path. Otherwise set `"image_url": null`.

Return valid JSON ONLY matching:
{{
  "document_title": "Document Title",
  "topics": [
    {{
      "name": "Topic Name",
      "summary": "Short summary",
      "flashcards": [
        {{
          "question": "Question text?",
          "answer": "Answer text.",
          "image_url": "/api/v1/documents/images/figure.png or null"
        }}
      ]
    }}
  ]
}}

STUDY MATERIAL CHUNK ({chunk_index + 1}/{total_chunks}):
{chunk}
"""
    async with semaphore:
        try:
            parsed_json = await generate_json_completion(
                prompt=prompt,
                system_prompt="You are an expert educational content generator. Return valid JSON only.",
                temperature=0.2,
                max_tokens=2500
            )
            if parsed_json:
                await set_cached_llm_result("flashcards_chunk", chunk, parsed_json, ttl=86400, model="groq-gemini")
            
            if progress_callback:
                try:
                    await progress_callback(chunk_index + 1, total_chunks)
                except Exception:
                    pass
            return parsed_json
        except Exception as e:
            logger.error(f"Error processing flashcard chunk {chunk_index + 1}: {e}")
            return None


def _generate_fallback_topics(chunks: List[str]) -> Dict[str, Any]:
    """Generates structural study topics directly from text if LLMs are unavailable."""
    full_text = " ".join(chunks)
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]

    headings = []
    for line in lines:
        if len(line) < 70 and (line.isupper() or line.istitle() or line.startswith(('#', '1.', '2.', '3.', 'Chapter', 'Section', 'Unit'))):
            clean = line.lstrip('#1234567890. -').strip()
            if clean and len(clean) > 3 and clean not in headings:
                headings.append(clean)
                if len(headings) >= 5:
                    break

    if not headings:
        headings = ["Core Concepts & Definitions", "Methodology & Architecture", "Key Revision Points"]

    topics = []
    for idx, h in enumerate(headings):
        start = idx * 6
        sample_slice = lines[start : start + 6]
        summary_text = " ".join(sample_slice) if sample_slice else f"Study section covering {h}."
        topics.append({
            "name": h,
            "summary": summary_text[:250],
            "flashcards": [
                {
                    "question": f"Explain the fundamental concept of {h}.",
                    "answer": summary_text[:200] or f"Key principles and theoretical foundations of {h} as detailed in the course material.",
                    "image_url": None
                },
                {
                    "question": f"What are the core applications and implications of {h}?",
                    "answer": f"Applies directly to syllabus problem-solving, architectural workflows, and examination evaluations.",
                    "image_url": None
                }
            ]
        })

    import re
    figure_urls = re.findall(r"\[Embedded Figure / Diagram:\s*(/api/v1/documents/images/[^\]\s]+)\]", full_text)
    if figure_urls and topics:
        for f_idx, fig_url in enumerate(figure_urls[:4]):
            topics[min(f_idx, len(topics) - 1)]["flashcards"].append({
                "question": f"Analyze and explain the diagram illustrated in Figure {f_idx + 1}.",
                "answer": "Visual diagram illustrating the core concepts and structural components of this topic.",
                "image_url": fig_url
            })

    return {
        "document_title": headings[0] if headings else "Study Document",
        "topics": topics
    }


async def generate_flashcards(
    chunks: List[str],
    progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
) -> Dict[str, Any]:
    """Generates structured topics and flashcards using parallel execution through the LLM Router."""
    final_result = {
        "document_title": "Extracted Study Material",
        "topics": []
    }
    topics_map = {}
    semaphore = asyncio.Semaphore(4)

    try:
        chunk_tasks = [
            _process_flashcard_chunk(chunk, i, len(chunks), semaphore, progress_callback)
            for i, chunk in enumerate(chunks)
        ]
        chunk_results = await asyncio.gather(*chunk_tasks)

        for i, parsed_json in enumerate(chunk_results):
            if not parsed_json:
                continue

            if "document_title" in parsed_json and i == 0 and parsed_json["document_title"]:
                final_result["document_title"] = parsed_json["document_title"]

            for topic in parsed_json.get("topics", []):
                t_name = topic.get("name", "General").strip()
                t_key = t_name.lower()

                if t_key not in topics_map:
                    topics_map[t_key] = {
                        "name": t_name,
                        "summary": topic.get("summary", ""),
                        "flashcards": []
                    }
                else:
                    if topic.get("summary") and topic["summary"] not in topics_map[t_key]["summary"]:
                        topics_map[t_key]["summary"] += " " + topic["summary"]

                topics_map[t_key]["flashcards"].extend(topic.get("flashcards", []))

        final_result["topics"] = list(topics_map.values())
    except Exception as e:
        logger.error(f"Error during parallel flashcard generation: {e}")

    if not final_result["topics"]:
        logger.warning("Generating structural fallback topics from text content.")
        fallback = _generate_fallback_topics(chunks)
        final_result["topics"] = fallback["topics"]
        if not final_result.get("document_title") or final_result["document_title"] == "Extracted Study Material":
            final_result["document_title"] = fallback["document_title"]

    return final_result


async def extract_past_paper_questions(chunk: str) -> List[Dict[str, Any]]:
    """Extracts individual questions from a past paper chunk."""
    prompt = f"""Extract all individual exam questions from this past paper text chunk.
For each question, identify its core concept, marks (if mentioned), type (short, long, mcq), and difficulty.

Output MUST be valid JSON:
{{
  "questions": [
    {{
      "original_text": "Question text...",
      "concept": "Core Concept",
      "marks": 10,
      "question_type": "long",
      "difficulty": "medium"
    }}
  ]
}}

Text Chunk:
{chunk}
"""
    try:
        data = await generate_json_completion(prompt=prompt, temperature=0.1)
        return data.get("questions", [])
    except Exception as e:
        logger.error(f"Error extracting past paper questions: {e}")
        return []


async def generate_new_question_from_context(concept: str, context: str, marks: int, difficulty: str) -> Dict[str, Any]:
    """Generates a novel question for a given concept based on study notes."""
    prompt = f"""Based on the provided study notes, generate ONE new, high-quality examination question testing the concept of "{concept}".
Marks: {marks}
Target Difficulty: {difficulty}

Study Notes Context:
{context}

Output MUST be valid JSON:
{{
  "question_text": "...",
  "suggested_answer": "...",
  "marks": {marks}
}}
"""
    try:
        return await generate_json_completion(prompt=prompt, temperature=0.3)
    except Exception as e:
        logger.error(f"Error generating question from context: {e}")
        return {}


async def generate_flowchart_graph(user_prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Generates a structured revision flowchart graph (nodes & edges) using LLM Router."""
    system_prompt = """You are an expert technical diagram designer. Create a structured revision flowchart diagram.
Return valid JSON ONLY matching:
{
  "title": "Short descriptive title (under 30 characters)",
  "summary": "One sentence summary describing the flowchart logic",
  "nodes": [
    {"id": "node_1", "label": "Short label (under 24 characters)", "type": "start", "color": "amber"},
    {"id": "node_2", "label": "Process step", "type": "process", "color": "blue"}
  ],
  "edges": [
    {"from": "node_1", "to": "node_2", "label": ""}
  ]
}
Node types: "start", "process", "decision", "end".
Node colors: "amber", "blue", "purple", "emerald", "rose".
Provide 4 to 8 logical nodes. All edges must match valid node ids."""

    prompt = f"USER PROMPT:\n{user_prompt.strip()}"
    try:
        parsed_json = await generate_json_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            model_override=model
        )

        raw_nodes = parsed_json.get("nodes", [])
        raw_edges = parsed_json.get("edges", [])
        valid_nodes = []
        node_ids = set()

        for idx, n in enumerate(raw_nodes):
            nid = str(n.get("id") or f"node_{idx + 1}")
            node_ids.add(nid)
            valid_nodes.append({
                "id": nid,
                "label": str(n.get("label") or f"Step {idx + 1}")[:28],
                "type": str(n.get("type") or "process").lower() if str(n.get("type", "")).lower() in ["start", "decision", "process", "end"] else "process",
                "color": str(n.get("color") or "blue").lower() if str(n.get("color", "")).lower() in ["amber", "blue", "purple", "emerald", "rose"] else "blue",
            })

        valid_edges = []
        for e in raw_edges:
            efrom = str(e.get("from", ""))
            eto = str(e.get("to", ""))
            if efrom in node_ids and eto in node_ids:
                valid_edges.append({
                    "from": efrom,
                    "to": eto,
                    "label": str(e.get("label") or "")[:20],
                })

        if len(valid_edges) == 0 and len(valid_nodes) > 1:
            for i in range(len(valid_nodes) - 1):
                valid_edges.append({
                    "from": valid_nodes[i]["id"],
                    "to": valid_nodes[i + 1]["id"],
                    "label": "",
                })

        return {
            "title": str(parsed_json.get("title") or "AI Flowchart")[:40],
            "summary": str(parsed_json.get("summary") or f"Generated {len(valid_nodes)} nodes"),
            "nodes": valid_nodes,
            "edges": valid_edges,
            "model": "groq-gemini"
        }
    except Exception as e:
        logger.error(f"Error generating flowchart graph: {e}")
        return {}
