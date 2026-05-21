"""
Application configuration using Pydantic Settings.
All settings are loaded from environment variables / .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "MentorIA"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- OpenRouter ---
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "google/gemini-2.0-flash-001"

    # --- LLM Parameters ---
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = "./chroma_data"

    # --- Uploads ---
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50

    # --- Chunking ---
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100

    # --- RAG ---
    TOP_K_RESULTS: int = 5

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        path = Path(self.UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_path(self) -> Path:
        path = Path(self.CHROMA_PERSIST_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Cached singleton for application settings."""
    return Settings()
