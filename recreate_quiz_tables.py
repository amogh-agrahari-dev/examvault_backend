import asyncio
from database import engine, Base
from models import quiz_models
# Import other models to ensure they are discovered by Base if not already
from models import auth_models, document_models, planner_models

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(quiz_models.QuizAnswer.__table__.drop, checkfirst=True)
        await conn.run_sync(quiz_models.QuizAttempt.__table__.drop, checkfirst=True)
        await conn.run_sync(quiz_models.QuizQuestion.__table__.drop, checkfirst=True)
        await conn.run_sync(quiz_models.Quiz.__table__.drop, checkfirst=True)
        
        await conn.run_sync(Base.metadata.create_all)
        print("Dropped and recreated quiz tables successfully.")

asyncio.run(main())
