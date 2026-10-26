import os
import io
import base64
import uuid
import logging
from PIL import Image
from typing import Optional, Tuple
from config import settings
from services.llm_router import generate_completion

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
EXTRACTED_IMAGES_DIR = os.path.join(UPLOAD_DIR, "extracted_images")
os.makedirs(EXTRACTED_IMAGES_DIR, exist_ok=True)


def is_important_image(image_bytes: bytes, min_dim: int = 120, min_bytes: int = 3500) -> bool:
    """
    Filters out decorative icons, bullet dots, tiny spacers, and single-color lines.
    Returns True only for diagrams, charts, figures, photos, and infographics.
    """
    if not image_bytes or len(image_bytes) < min_bytes:
        return False
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            if width < min_dim or height < min_dim:
                return False
            # Check extreme aspect ratio (e.g. 1px tall divider line)
            aspect = width / max(height, 1)
            if aspect > 15.0 or aspect < 0.07:
                return False
            return True
    except Exception:
        return False


def save_extracted_image(doc_id: uuid.UUID, image_bytes: bytes, index: int = 0, ext: str = "png") -> Tuple[str, str]:
    """
    Saves an extracted image to disk and returns (disk_path, public_relative_url).
    Standardizes image through PIL (converts CMYK/TIFF to clean RGB PNG/JPEG).
    """
    clean_ext = ext.lower().replace(".", "")
    if clean_ext not in {"png", "jpg", "jpeg", "webp"}:
        clean_ext = "png"
    
    filename = f"{doc_id}_fig_{index}.{clean_ext}"
    file_path = os.path.join(EXTRACTED_IMAGES_DIR, filename)

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(file_path, format="PNG" if clean_ext == "png" else "JPEG")
    except Exception:
        with open(file_path, "wb") as f:
            f.write(image_bytes)

    public_url = f"/api/v1/documents/images/{filename}"
    return file_path, public_url


async def analyze_image_with_vision(image_bytes: bytes, context_hint: str = "") -> Optional[str]:
    """
    Analyzes an educational diagram, chart, or figure using Google Gemini 1.5 Flash Vision.
    Transcribes text and explains diagrams, equations, or architectures.
    Cross-platform, production-ready on Linux/Render.
    """
    prompt = (
        "You are an academic document parser. Carefully examine this educational figure/diagram.\n"
        f"{'Context: ' + context_hint if context_hint else ''}\n"
        "1. Transcribe all readable text, labels, numbers, and formulas accurately.\n"
        "2. If this is a diagram, flowchart, chart, or architecture, concisely explain the flow and core concepts.\n"
        "Keep your output clear, academic, and directly educational."
    )

    try:
        result = await generate_completion(
            prompt=prompt,
            is_multimodal=True,
            image_bytes=image_bytes,
            temperature=0.1,
            max_tokens=600
        )
        return result.strip() if result else None
    except Exception as e:
        logger.warning(f"Error in analyze_image_with_vision: {e}")
        return None


async def transcribe_scanned_paper_image(image_bytes: bytes, page_hint: str = "") -> str:
    """
    High-accuracy cloud transcription of scanned hardcopy examination papers using Gemini 1.5 Flash Vision.
    Extracts exam titles, question numbers, sub-questions, marks, options, equations, and diagrams.
    Zero macOS/Apple Vision dependencies; runs cleanly on cloud/Docker.
    """
    prompt = (
        "You are an expert academic examination OCR transcriber.\n"
        f"{page_hint}\n"
        "Carefully read and transcribe every question from this scanned physical examination paper.\n"
        "Requirements:\n"
        "1. Transcribe the university/institution header, subject, time allowed, and maximum marks.\n"
        "2. Accurately transcribe all Question numbers (e.g. Q.1, Q.2), sub-questions (e.g. a, b, c), choices ('OR'), and marks in brackets (e.g. [7], [8]).\n"
        "3. Transcribe mathematical formulas, symbols, and technical terms verbatim.\n"
        "4. If there is a diagram or graph, briefly note '[Diagram: Description of diagram]'.\n"
        "Output clear, structured text reflecting the original question paper layout."
    )

    try:
        result = await generate_completion(
            prompt=prompt,
            is_multimodal=True,
            image_bytes=image_bytes,
            temperature=0.1,
            max_tokens=2000
        )
        return result.strip() if result else ""
    except Exception as e:
        logger.warning(f"Error transcribing scanned paper image: {e}")
        return ""
