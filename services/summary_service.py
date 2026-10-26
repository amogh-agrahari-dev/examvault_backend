import json
import logging
from typing import List, Dict, Any
from services.redis_service import get_cached_llm_result, set_cached_llm_result
from services.llm_router import generate_completion, generate_json_completion

logger = logging.getLogger(__name__)


async def organize_topics_with_llm(topics_data: List[dict]) -> Dict[str, Any]:
    """
    Takes a list of raw topic names and descriptions from multiple documents and 
    uses LLM to organize them into a deduplicated hierarchical tree.
    """
    raw_str = json.dumps(topics_data, sort_keys=True)
    cached = await get_cached_llm_result("summary_tree", raw_str, model="groq-gemini")
    if cached:
        logger.info("⚡ [Redis Cache HIT] Organized topics tree loaded in 1ms")
        return cached

    prompt = f"""You are an expert academic syllabus designer. I have a list of topics extracted from a student's study documents.
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
    parsed = await generate_json_completion(prompt=prompt, temperature=0.1)
    if parsed:
        await set_cached_llm_result("summary_tree", raw_str, parsed, ttl=86400, model="groq-gemini")
    return parsed


async def extract_relevant_facts(chunk: str, topic_name: str) -> str:
    """Extracts facts from a text chunk relevant to the given topic."""
    cached = await get_cached_llm_result("summary_facts", chunk, topic=topic_name, model="groq-gemini")
    if cached is not None:
        return cached

    prompt = f"""You are an engineering assistant. Your task is to extract information strictly relevant to the topic: "{topic_name}".
If the chunk contains relevant formulas, definitions, examples, or explanations, extract them clearly.
If the chunk has NO relevant information, output exactly the word "NONE".

Chunk:
{chunk}
"""
    text = await generate_completion(prompt=prompt, temperature=0.1)
    text = text.strip()
    result = "" if (text.upper() == "NONE" or not text) else text
    await set_cached_llm_result("summary_facts", chunk, result, ttl=86400, topic=topic_name, model="groq-gemini")
    return result


async def generate_final_topic_summary(topic_name: str, subtopics: List[str], intermediate_notes: str) -> Dict[str, Any]:
    """Generates the final structured topic summary based on the intermediate notes."""
    prompt = f"""You are an expert engineering professor. Write a structured study guide for the topic "{topic_name}".
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
    return await generate_json_completion(prompt=prompt, temperature=0.2, max_tokens=3500)
