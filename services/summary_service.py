import httpx
import json
import logging
from typing import List, Dict, Any, Callable, Awaitable
from config import settings

from services.redis_service import get_cached_llm_result, set_cached_llm_result

logger = logging.getLogger(__name__)

async def organize_topics_with_llm(topics_data: List[dict]) -> Dict[str, Any]:
    """
    Takes a list of raw topic names and descriptions from multiple documents and 
    uses LLM to organize them into a deduplicated hierarchical tree.
    """
    raw_str = json.dumps(topics_data, sort_keys=True)
    cached = await get_cached_llm_result("summary_tree", raw_str, model=settings.OLLAMA_MODEL)
    if cached:
        logger.info("⚡ [Redis Cache HIT] Organized topics tree loaded in 1ms")
        return cached

    prompt = f"""
You are an expert academic syllabus designer. I have a list of topics extracted from a student's study documents.
Your task is to merge duplicate topics, group them logically into high-level subjects/units, and organize them into a clean hierarchical JSON structure.

Raw Topics:
{json.dumps(topics_data, indent=2)}

You must return valid JSON ONLY, strictly adhering to this structure:
{{
  "tree": [
    {{
      "name": "High-level Unit or Topic",
      "subtopics": ["Subtopic 1", "Subtopic 2"],
      "source_topic_ids": ["uuid of related raw topics"]
    }}
  ]
}}
Do not generate anything other than JSON.
"""
    async with httpx.AsyncClient(timeout=120.0) as client:
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
        if not response_text:
            raise ValueError("Empty response from Ollama")
            
        parsed = None
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            if "```json" in response_text:
                clean_text = response_text.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(clean_text)
            else:
                raise ValueError("Failed to parse JSON")

        if parsed:
            await set_cached_llm_result("summary_tree", raw_str, parsed, ttl=86400, model=settings.OLLAMA_MODEL)
        return parsed


async def extract_relevant_facts(chunk: str, topic_name: str) -> str:
    """Extracts facts from a text chunk relevant to the given topic."""
    cached = await get_cached_llm_result("summary_facts", chunk, topic=topic_name, model=settings.OLLAMA_MODEL)
    if cached is not None:
        return cached

    prompt = f"""
You are an engineering assistant. Your task is to extract information strictly relevant to the topic: "{topic_name}".
If the chunk contains relevant formulas, definitions, examples, or explanations, extract them clearly.
If the chunk has NO relevant information, output exactly the word "NONE".

Chunk:
{chunk}
"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        result = "" if (text.upper() == "NONE" or not text) else text
        await set_cached_llm_result("summary_facts", chunk, result, ttl=86400, topic=topic_name, model=settings.OLLAMA_MODEL)
        return result



async def generate_final_topic_summary(topic_name: str, subtopics: List[str], intermediate_notes: str) -> Dict[str, Any]:
    """Generates the final structured topic summary based on the intermediate notes."""
    prompt = f"""
You are an expert engineering professor. Write a structured study guide for the topic "{topic_name}".
Include these subtopics if relevant: {', '.join(subtopics) if subtopics else 'N/A'}.

Use ONLY the provided Source Notes. Do not invent information.

Source Notes:
{intermediate_notes}

Return valid JSON ONLY with this schema:
{{
  "overview": "Clear overview and definition (use LaTeX $...$ for inline formulas)",
  "prerequisites": ["Concept 1", "Concept 2"],
  "core_concepts": [
    {{"heading": "Concept Name", "explanation": "Detailed explanation with LaTeX formulas where applicable"}}
  ],
  "formulas": [
    {{"equation_latex": "Pure LaTeX equation without dollar signs, e.g. E = mc^2", "variables": "LaTeX formatted variables e.g. $E$: Energy, $m$: mass", "explanation": "Physical/mathematical meaning"}}
  ],
  "algorithms_processes": "Step by step working if any",
  "examples": [
    {{"problem": "Problem statement with LaTeX math", "solution": "Step-by-step solution with LaTeX formulas"}}
  ],
  "revision_sheet": ["Key point 1 with LaTeX notation", "Key point 2"]
}}
"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2}
            }
        )
        response.raise_for_status()
        data = response.json()
        
        response_text = data.get("response", "")
        if not response_text:
            raise ValueError("Empty response from Ollama")
            
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            if "```json" in response_text:
                clean_text = response_text.split("```json")[1].split("```")[0].strip()
                return json.loads(clean_text)
            raise ValueError("Failed to parse JSON")
