from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import engine, Base
from models import auth_models, document_models, planner_models
from routes.auth import router as auth_router
from routes.documents import router as documents_router
from routes.planner import router as planner_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Automatically create tables in Neon on startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(planner_router)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}