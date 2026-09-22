import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from models.document_models import Document, Topic, Flashcard

async def save_document(
    db: AsyncSession,
    user_id: uuid.UUID,
    appwrite_file_id: str,
    filename: str,
    mime_type: str,
    file_size: int,
    status: str = "uploaded"
) -> Document:
    doc = Document(
        user_id=user_id,
        appwrite_file_id=appwrite_file_id,
        filename=filename,
        mime_type=mime_type,
        file_size=file_size,
        status=status
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc

async def save_generated_content(
    db: AsyncSession,
    document_id: uuid.UUID,
    user_id: uuid.UUID,
    generated_data: dict
):
    topics_data = generated_data.get("topics", [])
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
        
        flashcards_data = topic_item.get("flashcards", [])
        for f_idx, card_item in enumerate(flashcards_data):
            question = card_item.get("question")
            answer = card_item.get("answer")
            if not question or not answer:
                continue
                
            flashcard = Flashcard(
                document_id=document_id,
                topic_id=topic.id,
                user_id=user_id,
                question=question,
                answer=answer,
                order_index=f_idx
            )
            db.add(flashcard)
            
        await db.commit()
