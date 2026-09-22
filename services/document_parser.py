import os
import fitz  # PyMuPDF
import docx
import pptx

def extract_text(file_path: str, mime_type: str) -> str:
    """Extracts text from a given file based on its MIME type or extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf' or mime_type == 'application/pdf':
        return _extract_from_pdf(file_path)
    elif ext == '.docx' or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return _extract_from_docx(file_path)
    elif ext == '.pptx' or mime_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        return _extract_from_pptx(file_path)
    elif ext == '.txt' or mime_type == 'text/plain':
        return _extract_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext or mime_type}")

def chunk_text(text: str, max_words: int = 1500) -> list[str]:
    """Splits a large string of text into smaller chunks for semantic processing."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def _extract_from_pdf(file_path: str) -> str:
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text() + "\n\n"
        doc.close()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        raise
    return text.strip()

def _extract_from_docx(file_path: str) -> str:
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        raise
    return text.strip()

def _extract_from_pptx(file_path: str) -> str:
    text = ""
    try:
        prs = pptx.Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text += shape.text + "\n"
            text += "\n" # separate slides
    except Exception as e:
        print(f"Error reading PPTX: {e}")
        raise
    return text.strip()

def _extract_from_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # Fallback for some windows encodings
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read().strip()
