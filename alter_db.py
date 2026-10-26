import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from config import settings
from sqlalchemy import text

async def main():
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as conn:
        # First, ensure subjects table exists before foreign key is added
        from models import subject_models
        await conn.run_sync(subject_models.Base.metadata.create_all)
        
        try:
            await conn.execute(text("ALTER TABLE documents ADD COLUMN subject_id UUID;"))
        except Exception as e:
            print("subject_id exists?", e)
        try:
            await conn.execute(text("ALTER TABLE documents ADD COLUMN document_type VARCHAR(50) DEFAULT 'NOTES';"))
        except Exception as e:
            print("document_type exists?", e)
        try:
            await conn.execute(text("ALTER TABLE documents ADD COLUMN year INTEGER;"))
        except Exception as e:
            print("year exists?", e)
            
        try:
            await conn.execute(text("ALTER TABLE documents ADD CONSTRAINT fk_documents_subject FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE SET NULL;"))
        except Exception as e:
            print("fk exists?", e)
            
    await engine.dispose()
    print("Done")

if __name__ == "__main__":
    asyncio.run(main())
