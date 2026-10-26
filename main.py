from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.middleware.gzip import GZipMiddleware

from config import settings
from database import engine, Base
from services.redis_service import init_redis, close_redis, is_redis_available
from models import auth_models, document_models, planner_models, quiz_models, summary_models, chat_models, discussion_models, subject_models, note_models, past_paper_models
from routes.auth import router as auth_router
from routes.documents import router as documents_router
from routes.planner import router as planner_router
from routes.quiz import router as quiz_router
from routes.summary import router as summary_router
from routes.chat import router as chat_router
from routes.discussions import router as discussions_router
from routes.subjects import router as subjects_router
from routes.predicted_papers import router as predicted_papers_router
from routes.notes import router as notes_router
from routes.users import router as users_router
from routes.past_papers import router as past_papers_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Redis connection pool
    await init_redis()
    # Automatically create tables in Neon on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    try:
        await close_redis()
    except Exception:
        pass
    try:
        await engine.dispose()
    except Exception:
        pass

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Response compression: shrink large JSON (quizzes, notes, topics) by ~75%
app.add_middleware(GZipMiddleware, minimum_size=1000)

allowed_origins = [
    "https://examvault-two.vercel.app",
    "https://examvault.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(planner_router)
app.include_router(quiz_router)
app.include_router(summary_router)
app.include_router(chat_router)
app.include_router(discussions_router)
app.include_router(subjects_router)
app.include_router(predicted_papers_router)
app.include_router(notes_router)
app.include_router(users_router)
app.include_router(past_papers_router)

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}

@app.api_route("/api/v1/health", methods=["GET", "HEAD"])
async def health():
    return {
        "status": "healthy",
        "redis_available": is_redis_available(),
        "database_configured": bool(settings.DATABASE_URL)
    }