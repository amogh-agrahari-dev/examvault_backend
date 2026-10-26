"""
Legacy Ollama service proxy.
All LLM generation is now routed through the cloud-based dual-provider architecture
(Groq + Google Gemini) in services.llm_router.
This module is maintained for backward compatibility.
"""
from services.llm_router import (
    generate_completion,
    generate_json_completion,
    generate_flashcards,
    extract_past_paper_questions,
    generate_new_question_from_context,
    generate_flowchart_graph,
    _process_flashcard_chunk,
    _generate_fallback_topics
)

__all__ = [
    "generate_completion",
    "generate_json_completion",
    "generate_flashcards",
    "extract_past_paper_questions",
    "generate_new_question_from_context",
    "generate_flowchart_graph",
    "_process_flashcard_chunk",
    "_generate_fallback_topics"
]
