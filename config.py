import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "ExamVault API"
    DATABASE_URL: str
    JWT_SECRET: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Redis config
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    REDIS_ENABLED: bool = True
    REDIS_CACHE_TTL: int = 1800  # 30 mins

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        extra="ignore"
    )

    # Appwrite config
    APPWRITE_ENDPOINT: str
    APPWRITE_PROJECT_ID: str
    APPWRITE_API_KEY: str
    APPWRITE_BUCKET_ID: str

    # Gemini Config
    GEMINI_API_KEY: str

    # Ollama config
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma2:9b"

    # Code Execution / Sandbox Config (optional, defaults to local isolated sandbox)
    JUDGE0_API_URL: str = ""
    JUDGE0_API_KEY: str = ""
    SANDBOX_TIMEOUT_SECONDS: float = 7.0

settings = Settings()