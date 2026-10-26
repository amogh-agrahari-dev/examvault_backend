import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import httpx
import json
from config import settings
from services.llm_router import generate_completion, generate_json_completion

logger = logging.getLogger(__name__)

TOP_K_CHUNKS = 5
MIN_RELEVANCE_SCORE = 0.05
MAX_CONTEXT_CHARS = 8000
MAX_HISTORY_MESSAGES = 6  # last N messages to include in prompt


def _tokenize(text: str) -> List[str]:
    """Simple word tokenizer — lowercases and strips punctuation."""
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    # Remove very short tokens and common stop-words
    STOPWORDS = {
        "the","a","an","is","it","in","of","to","and","or","for","on","with",
        "that","this","be","are","was","has","have","at","by","from","as",
        "but","not","we","he","she","they","i","you","can","will","do","did",
        "if","up","out","about","which","when","what","how","all","been"
    }
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def _compute_tf(tokens: List[str]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {w: c / total for w, c in counts.items()}


def _bm25_score(query_tokens: List[str], doc_tokens: List[str], k1: float = 1.5, b: float = 0.75, avg_len: float = 200.0) -> float:
    """
    BM25 scoring between a query and a document token list.
    Used to rank document chunks by keyword relevance.
    """
    doc_freq = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 0.0
    for token in query_tokens:
        if token not in doc_freq:
            continue
        tf = doc_freq[token]
        idf = 1.0  # simplified IDF = 1 since we operate per-query
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
        score += idf * (numerator / denominator)
    return score


def retrieve_relevant_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = TOP_K_CHUNKS,
    min_score: float = MIN_RELEVANCE_SCORE
) -> List[Dict[str, Any]]:
    """
    Retrieves the most relevant chunks for a query using BM25 keyword scoring.
    Each chunk dict must have keys: id, content, document_id, document_name, chunk_index, topic_id.
    Returns top_k chunks with a 'relevance' field added, sorted descending.
    """
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return chunks[:top_k]

    # Compute average doc length for BM25 normalisation
    all_token_lists = [_tokenize(c["content"]) for c in chunks]
    avg_len = sum(len(t) for t in all_token_lists) / len(all_token_lists) if all_token_lists else 200.0

    scored = []
    for chunk, tokens in zip(chunks, all_token_lists):
        score = _bm25_score(query_tokens, tokens, avg_len=avg_len)
        scored.append({**chunk, "relevance": score})

    # Sort descending
    scored.sort(key=lambda x: x["relevance"], reverse=True)

    # Filter by minimum score threshold
    filtered = [c for c in scored if c["relevance"] >= min_score]
    return filtered[:top_k] if filtered else scored[:top_k]


def build_context_string(retrieved_chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, List[Dict]]:
    """
    Builds an LLM-ready context string from retrieved chunks.
    Returns (context_str, sources_list).
    """
    context_parts = []
    sources = []
    used_chars = 0

    for i, chunk in enumerate(retrieved_chunks):
        doc_name = chunk.get("document_name", "Unknown Document")
        chunk_idx = chunk.get("chunk_index", 0)
        content = chunk.get("content", "").strip()

        if not content:
            continue

        entry = f"[Source {i+1}: {doc_name} — Section {chunk_idx + 1}]\n{content}"
        if used_chars + len(entry) > max_chars:
            # Truncate to fit
            available = max_chars - used_chars - len(f"[Source {i+1}: {doc_name} — Section {chunk_idx + 1}]\n") - 20
            if available > 100:
                entry = f"[Source {i+1}: {doc_name} — Section {chunk_idx + 1}]\n{content[:available]}..."
            else:
                break

        context_parts.append(entry)
        sources.append({
            "document_id": str(chunk.get("document_id", "")),
            "document_name": doc_name,
            "chunk_id": str(chunk.get("id", "")),
            "chunk_index": chunk_idx,
            "relevance": round(chunk.get("relevance", 0.0), 4)
        })
        used_chars += len(entry)

    return "\n\n---\n\n".join(context_parts), sources


def build_chat_prompt(
    subject_name: str,
    topic_name: str,
    context_str: str,
    conversation_history: List[Dict[str, str]],
    user_question: str
) -> str:
    """
    Builds the final grounded prompt sent to the LLM.
    Strict instruction: answer ONLY from the provided document context.
    """
    history_str = ""
    if conversation_history:
        history_parts = []
        for msg in conversation_history[-MAX_HISTORY_MESSAGES:]:
            role = "Student" if msg["role"] == "user" else "AI Assistant"
            history_parts.append(f"{role}: {msg['content']}")
        history_str = "\n".join(history_parts)

    no_context_instruction = (
        "If the provided document context does not contain enough information to answer the question, "
        "you MUST respond EXACTLY with: "
        "'I couldn't find enough information about this in the uploaded document \"{subject_name}\". "
        "Please ensure relevant study material has been uploaded for this topic.' "
        "Do NOT use your own general knowledge to answer. "
        "Do NOT hallucinate or invent facts."
    ).format(subject_name=subject_name)

    prompt = f"""You are an AI study assistant for engineering students. You help students understand study material they have uploaded.

The student is currently studying:
Document/Subject: {subject_name}
Topic: {topic_name}

STRICT RULES — you MUST follow these without exception:
1. Answer ONLY using the DOCUMENT CONTEXT provided below.
2. Do NOT use general knowledge or information not present in the context.
3. If the context does not contain sufficient information to answer: {no_context_instruction}
4. Never fabricate definitions, formulas, or facts.
5. Preserve important terminology and formulas from the source material. Format all mathematical formulas, variables, and equations in standard LaTeX notation: use $...$ for inline expressions (e.g. $E = mc^2$, $O(n \\log n)$) and $$...$$ on separate lines for block equations.
6. When answering, mention which source/section the information came from (e.g., "According to Source 1...").
7. Structure your answer clearly for a student: use definitions, explanations, and examples where the documents support it.
8. Keep your answer focused on the topic: {topic_name}. Do not drift into other topics.

DOCUMENT CONTEXT:
{context_str if context_str.strip() else "[NO RELEVANT DOCUMENT CONTENT FOUND FOR THIS QUESTION]"}

{"CONVERSATION HISTORY (for context only — still answer using documents above):" + chr(10) + history_str if history_str else ""}

STUDENT QUESTION: {user_question}

ANSWER:"""

    return prompt


async def call_ollama_for_chat(prompt: str) -> str:
    """
    Calls the cloud dual-provider LLM router (Groq/Gemini) with grounded prompt.
    Returns the text response.
    """
    try:
        return await generate_completion(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2500
        )
    except Exception as e:
        logger.error(f"Error in call_ollama_for_chat: {e}")
        return "I am currently unable to process your request due to an AI service error. Please try again shortly."


def build_verification_prompt(claim: str, context_str: str) -> str:
    """
    Builds the verification prompt for a user claim against topic context.
    Expects a JSON output.
    """
    return f"""You are an academic claim verification assistant.

Your task is to evaluate the user's claim ONLY against the provided evidence retrieved from the selected topic.

Possible verdicts:
SUPPORTED: The evidence directly or strongly supports the claim.
CONTRADICTED: The evidence directly conflicts with the claim.
INSUFFICIENT_EVIDENCE: The provided evidence does not contain enough information to determine whether the claim is true or false.

Rules:
1. Do not invent evidence.
2. Do not fabricate citations.
3. Do not treat missing evidence as contradiction.
4. Do not use unrelated topics or general internet knowledge.
5. Do not claim that the uploaded documents say something they do not say.
6. Clearly distinguish evidence from inference.
7. Keep the explanation concise but academically useful.
8. If evidence is insufficient, explicitly say so in the explanation.

DOCUMENT EVIDENCE:
{context_str if context_str.strip() else "[NO RELEVANT DOCUMENT CONTENT FOUND FOR THIS CLAIM]"}

CLAIM TO VERIFY:
"{claim}"

You MUST respond strictly with valid JSON matching exactly this schema:
{{
  "verdict": "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE",
  "explanation": "A clear, concise explanation of why this verdict was reached based ONLY on the evidence."
}}
"""

async def call_ollama_for_verification(prompt: str) -> dict:
    """
    Calls cloud dual-provider LLM router enforcing JSON format and returns parsed dict.
    """
    try:
        return await generate_json_completion(
            prompt=prompt,
            temperature=0.0,
            max_tokens=1000
        )
    except Exception as e:
        logger.error(f"Error in call_ollama_for_verification: {e}")
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "explanation": "Could not parse AI response or verification service is unavailable."
        }
