import httpx
import json
import logging
from typing import Dict, Any, List, Callable, Awaitable
from config import settings

logger = logging.getLogger(__name__)

async def generate_flashcards(chunks: List[str], progress_callback: Callable[[int, int], Awaitable[None]] = None) -> Dict[str, Any]:
    """
    Sends document text chunks to Ollama and aggregates the results.
    Uses an advanced prompt to enforce exhaustive extraction and high quality flashcards.
    """
    final_result = {
        "document_title": "Extracted Study Material",
        "topics": []
    }
    
    # We will map topics by name (lowercased) to merge them across chunks
    topics_map = {}

    for i, chunk in enumerate(chunks):
        if progress_callback:
            await progress_callback(i + 1, len(chunks))
            
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        prompt = f"""
You are an expert academic tutor and instructional designer. Your task is to process the following chunk of study material and meticulously convert it into high-yield revision flashcards.

Your goals:
1. **Be Exhaustive**: Do not leave any significant concept, definition, formula, or key fact behind. 
2. **Be Conceptual**: Flashcards should test deep understanding, comparisons, and mechanisms, not just rote facts.
3. **Be Structured**: Group the flashcards into logical, high-level "topics". Create as many topics and flashcards as necessary to cover the material completely.
4. **Be Strict**: Do not generate flashcards on meaningless boilerplate like table of contents, headers, footers, or bibliography. 

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
          "answer": "Answer text."
        }}
      ]
    }}
  ]
}}

STUDY MATERIAL CHUNK ({i+1}/{len(chunks)}):
{chunk}
"""

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "prompt": prompt,
                        "format": "json",
                        "stream": False,
                        "options": {
                            "temperature": 0.2
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                response_text = data.get("response", "")
                if not response_text:
                    logger.warning(f"Empty response from Ollama for chunk {i+1}")
                    continue
                    
                # Parse JSON
                try:
                    parsed_json = json.loads(response_text)
                except json.JSONDecodeError:
                    if "```json" in response_text:
                        clean_text = response_text.split("```json")[1].split("```")[0].strip()
                        parsed_json = json.loads(clean_text)
                    else:
                        logger.warning(f"Failed to parse JSON for chunk {i+1}")
                        continue
                        
                # Merge logic
                if "document_title" in parsed_json and i == 0:
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
                        # Append to existing summary if new summary provided
                        if topic.get("summary") and topic["summary"] not in topics_map[t_key]["summary"]:
                            topics_map[t_key]["summary"] += " " + topic["summary"]
                            
                    topics_map[t_key]["flashcards"].extend(topic.get("flashcards", []))
                    
        except Exception as e:
            logger.error(f"Error communicating with Ollama on chunk {i+1}: {e}")
            # Continue to next chunk instead of entirely failing
            continue
            
    final_result["topics"] = list(topics_map.values())
    
    if not final_result["topics"]:
        raise ValueError("Failed to generate any valid flashcards from the document chunks.")
        
    return final_result
