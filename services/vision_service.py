import os
import io
import base64
import uuid
import logging
import httpx
from PIL import Image
from typing import Optional, Tuple
from config import settings

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
EXTRACTED_IMAGES_DIR = os.path.join(UPLOAD_DIR, "extracted_images")
os.makedirs(EXTRACTED_IMAGES_DIR, exist_ok=True)

# Cache discovered vision model
_VISION_MODEL: Optional[str] = None
_VISION_CHECKED: bool = False


async def get_vision_model() -> Optional[str]:
    """Detects available vision-capable model in local Ollama instance."""
    global _VISION_MODEL, _VISION_CHECKED
    if _VISION_CHECKED:
        return _VISION_MODEL

    known_vision_models = ["gemma3:12b", "llama3.2-vision", "llava", "bakllava", "moondream", "minicpm-v"]
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            res = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if res.status_code == 200:
                data = res.json()
                models = data.get("models", [])
                
                # First check models that advertise vision capabilities
                for m in models:
                    caps = m.get("capabilities", [])
                    if "vision" in caps:
                        _VISION_MODEL = m.get("name")
                        logger.info(f"Discovered vision model with capability tag: {_VISION_MODEL}")
                        _VISION_CHECKED = True
                        return _VISION_MODEL

                # Check by known names
                for m in models:
                    name = m.get("name", "")
                    for kvm in known_vision_models:
                        if kvm in name:
                            _VISION_MODEL = name
                            logger.info(f"Discovered known vision model: {_VISION_MODEL}")
                            _VISION_CHECKED = True
                            return _VISION_MODEL
    except Exception as e:
        logger.warning(f"Failed to query Ollama for vision models: {e}")

    _VISION_CHECKED = True
    return _VISION_MODEL


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
    """
    clean_ext = ext.lower().replace(".", "")
    if clean_ext not in {"png", "jpg", "jpeg", "webp"}:
        clean_ext = "png"
    
    filename = f"{doc_id}_fig_{index}.{clean_ext}"
    file_path = os.path.join(EXTRACTED_IMAGES_DIR, filename)

    # Standardize image through PIL (converts CMYK/TIFF to clean RGB PNG/JPEG if needed)
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
    Analyzes an image using Ollama vision model.
    Transcribes text and explains diagrams, equations, or tables.
    """
    vision_model = await get_vision_model()
    if not vision_model:
        return None

    try:
        # Resize huge images to max 1024x1024 to speed up local inference
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "You are an academic document parser. Carefully examine this educational figure/diagram.\n"
            f"{'Context: ' + context_hint if context_hint else ''}\n"
            "1. Transcribe all readable text, labels, numbers, and formulas accurately.\n"
            "2. If this is a diagram, flowchart, chart, or architecture, concisely explain the flow and core concepts.\n"
            "Keep your output clear, academic, and directly educational."
        )

        async with httpx.AsyncClient(timeout=35.0) as client:
            res = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": vision_model,
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 300
                    }
                }
            )
            if res.status_code == 200:
                data = res.json()
                analysis = data.get("response", "").strip()
                if analysis:
                    return analysis
    except Exception as e:
        logger.warning(f"Error in analyze_image_with_vision: {e}")

    return None


async def transcribe_scanned_paper_image(image_bytes: bytes, page_hint: str = "") -> str:
    """
    High-accuracy transcription of scanned hardcopy examination papers using vision AI.
    Extracts exam titles, question numbers, sub-questions, marks, options, equations, and diagrams.
    """
    vision_model = await get_vision_model()
    if not vision_model:
        return ""

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            # Preserve high resolution for scanned paper text and formulas
            img.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

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

        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": vision_model,
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1200
                    }
                }
            )
            if res.status_code == 200:
                data = res.json()
                return data.get("response", "").strip()
    except Exception as e:
        logger.warning(f"Error transcribing scanned paper image: {e}")

    return ""
