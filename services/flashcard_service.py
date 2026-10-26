import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from models.document_models import Document, Topic, Flashcard
from models.chat_models import DocumentChunk

async def save_document(
    db: AsyncSession,
    user_id: uuid.UUID,
    appwrite_file_id: str,
    filename: str,
    mime_type: str,
    file_size: int,
    status: str = "uploaded",
    subject_id: uuid.UUID = None,
    document_type: str = "NOTES",
    year: int = None
) -> Document:
    doc = Document(
        user_id=user_id,
        appwrite_file_id=appwrite_file_id,
        filename=filename,
        mime_type=mime_type,
        file_size=file_size,
        status=status,
        subject_id=subject_id,
        document_type=document_type,
        year=year
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc

async def save_generated_content(
    db: AsyncSession,
    document_id: uuid.UUID,
    user_id: uuid.UUID,
    generated_data: dict,
    raw_chunks: list = None
):
    """
    Saves AI-generated topics, flashcards, and document chunks to the database.
    raw_chunks: the original text chunks from chunk_text(), used to populate DocumentChunk
                for the RAG chat retrieval system.
    """
    topics_data = generated_data.get("topics", [])
    saved_topics = []

    for t_idx, topic_item in enumerate(topics_data):
        topic_name = topic_item.get("name", "Untitled Topic")
        if not topic_name:
            continue
            
        topic = Topic(
            document_id=document_id,
            user_id=user_id,
            name=topic_name,
            summary=topic_item.get("summary", ""),
            order_index=t_idx
        )
        db.add(topic)
        await db.commit()
        await db.refresh(topic)
        saved_topics.append(topic)
        
        flashcards_data = topic_item.get("flashcards", [])
        for f_idx, card_item in enumerate(flashcards_data):
            question = card_item.get("question")
            answer = card_item.get("answer")
            if not question or not answer:
                continue
                
            image_url = card_item.get("image_url")
            if image_url and isinstance(image_url, str):
                import re
                match = re.search(r"/api/v1/documents/images/[\w\-.]+", image_url)
                image_url = match.group(0) if match else (image_url if image_url.startswith("/") else None)
            else:
                image_url = None

            flashcard = Flashcard(
                document_id=document_id,
                topic_id=topic.id,
                user_id=user_id,
                question=question,
                answer=answer,
                order_index=f_idx,
                image_url=image_url
            )
            db.add(flashcard)
            
        await db.commit()

    # ── Store document chunks for the RAG chat system ──────────────────────
    # Each raw text chunk is associated with the nearest topic (by chunk index
    # proportional to the number of topics). This gives the chat system
    # topic-scoped retrieval without requiring a vector database.
    if raw_chunks and saved_topics:
        num_chunks = len(raw_chunks)
        num_topics = len(saved_topics)
        for chunk_idx, chunk_text_content in enumerate(raw_chunks):
            if not chunk_text_content.strip():
                continue
            # Assign chunk to topic proportionally by position in the document
            topic_idx = min(int(chunk_idx / num_chunks * num_topics), num_topics - 1)
            assigned_topic = saved_topics[topic_idx]

            chunk = DocumentChunk(
                document_id=document_id,
                topic_id=assigned_topic.id,
                user_id=user_id,
                chunk_index=chunk_idx,
                content=chunk_text_content.strip()
            )
            db.add(chunk)
        await db.commit()
    elif raw_chunks:
        # No topics were found but we still have chunks — store without topic assignment
        for chunk_idx, chunk_text_content in enumerate(raw_chunks):
            if not chunk_text_content.strip():
                continue
            chunk = DocumentChunk(
                document_id=document_id,
                topic_id=None,
                user_id=user_id,
                chunk_index=chunk_idx,
                content=chunk_text_content.strip()
            )
            db.add(chunk)
        await db.commit()
