import asyncio
import httpx
import json
import logging
from typing import Dict, Any, List, Callable, Awaitable, Optional
from config import settings
from services.redis_service import get_cached_llm_result, set_cached_llm_result

logger = logging.getLogger(__name__)


async def _process_flashcard_chunk(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    semaphore: asyncio.Semaphore,
    progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None
) -> Optional[Dict[str, Any]]:
    """Processes a single text chunk with Redis caching and concurrency throttling."""
    # 1. Content-addressed Redis cache check
    cached = await get_cached_llm_result("flashcards_chunk", chunk, model=settings.OLLAMA_MODEL)
    if cached:
        logger.info(f"⚡ [Redis Cache HIT] Flashcards for chunk {chunk_index + 1}/{total_chunks}")
        if progress_callback:
            try:
                await progress_callback(chunk_index + 1, total_chunks)
            except Exception:
                pass
        return cached

    prompt = f"""
You are an expert academic tutor and instructional designer. Your task is to process the following chunk of study material and meticulously convert it into high-yield revision flashcards.

Your goals:
1. **Be Exhaustive**: Do not leave any significant concept, definition, formula, or key fact behind. 
2. **Be Conceptual**: Flashcards should test deep understanding, comparisons, and mechanisms, not just rote facts.
3. **Be Structured**: Group the flashcards into logical, high-level "topics". Create as many topics and flashcards as necessary to cover the material completely.
4. **Be Strict**: Do not generate flashcards on meaningless boilerplate like table of contents, headers, footers, or bibliography. 
5. **Use LaTeX for Math**: Format any equations, formulas, scientific laws, or mathematical symbols in clean LaTeX ($...$ or $$...$$). 
6. **Include Diagrams & Images**: If the chunk contains an `[Embedded Figure / Diagram: /api/v1/documents/images/...]`, create high-yield flashcards specifically testing the diagram, flowchart, or architecture! Set `"image_url": "/api/v1/documents/images/..."` with the exact URL from the chunk on that flashcard. For standard text-only flashcards without an image, set `"image_url": null`.

You must return valid JSON ONLY, strictly adhering to the following structure:
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
          "image_url": "/api/v1/documents/images/filename.png or null"
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
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 1000
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()

                response_text = data.get("response", "")
                if not response_text:
                    logger.warning(f"Empty response from Ollama for chunk {chunk_index + 1}")
                    return None

                parsed_json = None
                try:
                    parsed_json = json.loads(response_text)
                except json.JSONDecodeError:
                    if "```json" in response_text:
                        clean_text = response_text.split("```json")[1].split("```")[0].strip()
                        parsed_json = json.loads(clean_text)
                    elif "{" in response_text and "}" in response_text:
                        s_idx = response_text.find("{")
                        e_idx = response_text.rfind("}") + 1
                        parsed_json = json.loads(response_text[s_idx:e_idx])

                if parsed_json:
                    # Cache in Redis with 24h TTL
                    await set_cached_llm_result(
                        "flashcards_chunk", chunk, parsed_json, ttl=86400, model=settings.OLLAMA_MODEL
                    )

                if progress_callback:
                    try:
                        await progress_callback(chunk_index + 1, total_chunks)
                    except Exception:
                        pass

                return parsed_json
        except Exception as e:
            logger.error(f"Error communicating with Ollama on chunk {chunk_index + 1}: {e}")
            return None


def _generate_fallback_topics(chunks: List[str]) -> Dict[str, Any]:
    """Generates structural study topics and flashcards directly from text if LLM is unavailable."""
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
                    "answer": summary_text[:200] or f"Key principles and theoretical foundations of {h} as detailed in the course material."
                },
                {
                    "question": f"What are the core applications and implications of {h}?",
                    "answer": f"Applies directly to syllabus problem-solving, architectural workflows, and examination evaluations."
                }
            ]
        })

    # Extract any embedded diagram URLs from the text
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


async def generate_flashcards(chunks: List[str], progress_callback: Callable[[int, int], Awaitable[None]] = None) -> Dict[str, Any]:
    """
    Sends document text chunks to Ollama in parallel bounded by a Semaphore,
    leveraging Redis content-addressed cache for instant retrieval, with guaranteed fallback.
    """
    final_result = {
        "document_title": "Extracted Study Material",
        "topics": []
    }
    
    topics_map = {}

    # Bound concurrency to 2 parallel tasks to protect local compute while cutting runtime by ~50%
    semaphore = asyncio.Semaphore(2)

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
        logger.error(f"Error during parallel chunk generation: {e}")

    # If Ollama didn't return valid topics, generate clean structural fallback topics
    if not final_result["topics"]:
        logger.warning("Generating structural fallback topics from text content.")
        fallback = _generate_fallback_topics(chunks)
        final_result["topics"] = fallback["topics"]
        if not final_result.get("document_title") or final_result["document_title"] == "Extracted Study Material":
            final_result["document_title"] = fallback["document_title"]
        
    return final_result

async def extract_past_paper_questions(chunk: str) -> List[Dict[str, Any]]:
    """
    Extracts individual questions from a past paper chunk.
    """
    prompt = f"""
You are an expert exam analyzer. Extract all individual exam questions from the provided text chunk of a past examination paper.
For each question, try to identify its core concept, marks (if mentioned), type (short, long, mcq), and difficulty.

Output MUST be valid JSON in this exact structure:
{{
  "questions": [
    {{
      "original_text": "Describe the OSI model layers.",
      "concept": "OSI Model",
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.1}
                }
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "")
            
            try:
                parsed_json = json.loads(response_text)
            except json.JSONDecodeError:
                if "```json" in response_text:
                    clean_text = response_text.split("```json")[1].split("```")[0].strip()
                    parsed_json = json.loads(clean_text)
                else:
                    return []
            
            return parsed_json.get("questions", [])
    except Exception as e:
        logger.error(f"Error extracting past paper questions: {e}")
        return []

async def generate_new_question_from_context(concept: str, context: str, marks: int, difficulty: str) -> Dict[str, Any]:
    """
    Generates a NEW question for a given concept based on the provided context (study notes).
    """
    prompt = f"""
You are an expert exam paper setter. Based on the provided study notes, generate ONE new, high-quality examination question testing the concept of "{concept}".
Do NOT copy a question from a past paper. Create a fresh question.
Marks: {marks}
Target Difficulty: {difficulty}

Study Notes Context:
{context}

Output MUST be valid JSON in this structure:
{{
  "question_text": "...",
  "suggested_answer": "...",
  "marks": {marks}
}}
"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.4}
                }
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "")
            
            try:
                parsed_json = json.loads(response_text)
                return parsed_json
            except json.JSONDecodeError:
                if "```json" in response_text:
                    clean_text = response_text.split("```json")[1].split("```")[0].strip()
                    return json.loads(clean_text)
                return {}
    except Exception as e:
        logger.error(f"Error generating new question: {e}")
        return {}


async def generate_flowchart_graph(user_prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Generates a structured revision flowchart graph (nodes & edges) using Ollama.
    """
    selected_model = model or settings.OLLAMA_MODEL or "gemma2:9b"
    logger.info(f"Generating flowchart with Ollama model '{selected_model}' for prompt: {user_prompt[:60]}...")
    
    system_prompt = f"""You are an expert educational and technical diagram designer.
Create a structured revision flowchart diagram based on the user's prompt.

Output valid JSON ONLY strictly conforming to this schema:
{{
  "title": "Short descriptive title (under 30 characters)",
  "summary": "One sentence summary describing the flowchart logic",
  "nodes": [
    {{
      "id": "node_1",
      "label": "Short label (under 24 characters)",
      "type": "start",
      "color": "amber"
    }},
    {{
      "id": "node_2",
      "label": "Process step",
      "type": "process",
      "color": "blue"
    }}
  ],
  "edges": [
    {{
      "from": "node_1",
      "to": "node_2",
      "label": ""
    }}
  ]
}}

Rules:
1. Node types must only be: "start" (initial pill), "process" (rectangle task), "decision" (diamond condition), or "end" (terminal pill).
2. Node colors must only be one of: "amber", "blue", "purple", "emerald", "rose".
3. Provide 4 to 8 logical, well-ordered nodes. Keep all labels under 24 characters so they fit cleanly in boxes.
4. For decision nodes, connect outgoing edges with labels like "Yes" and "No".
5. Every edge "from" and "to" must match a valid node "id".
6. Return valid JSON only, without any commentary or markdown quotes.

USER PROMPT:
{user_prompt.strip()}
"""
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": selected_model,
                    "prompt": system_prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.3}
                }
            )
            response.raise_for_status()
            data = response.json()
            response_text = (data.get("response") or "").strip()
            
            try:
                parsed_json = json.loads(response_text)
            except json.JSONDecodeError:
                if "```json" in response_text:
                    clean_text = response_text.split("```json")[1].split("```")[0].strip()
                    parsed_json = json.loads(clean_text)
                elif "{" in response_text and "}" in response_text:
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}") + 1
                    parsed_json = json.loads(response_text[start_idx:end_idx])
                else:
                    raise ValueError("Could not parse JSON from Ollama response")

            # Validate nodes structure
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
                    "type": str(n.get("type") or "process").lower() if str(n.get("type", "")).lower() in ["start", "decision", "process", "end", "text"] else "process",
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

            # Ensure at least edges connect sequentially if LLM returned 0 edges
            if len(valid_edges) == 0 and len(valid_nodes) > 1:
                for i in range(len(valid_nodes) - 1):
                    valid_edges.append({
                        "from": valid_nodes[i]["id"],
                        "to": valid_nodes[i + 1]["id"],
                        "label": "",
                    })

            return {
                "title": str(parsed_json.get("title") or "AI Flowchart")[:40],
                "summary": str(parsed_json.get("summary") or f"Generated {len(valid_nodes)} nodes with Ollama ({selected_model})"),
                "nodes": valid_nodes,
                "edges": valid_edges,
                "model": selected_model,
            }

    except Exception as e:
        logger.error(f"Error generating flowchart with Ollama: {e}")
        return {}
