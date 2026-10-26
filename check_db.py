import asyncio
from sqlalchemy.future import select
from database import AsyncSessionLocal
from models.document_models import Topic, Document
from models.chat_models import DocumentChunk

async def check_db():
    async with AsyncSessionLocal() as db:
        docs = await db.execute(select(Document))
        all_docs = docs.scalars().all()
        for doc in all_docs:
            print(f"Doc: {doc.filename} (ID: {doc.id}, User: {doc.user_id})")
        
        chunks_res = await db.execute(select(DocumentChunk))
        all_chunks = chunks_res.scalars().all()
        for c in all_chunks:
            print(f"Chunk Doc ID: {c.document_id}, User ID: {c.user_id}")

if __name__ == "__main__":
    asyncio.run(check_db())
