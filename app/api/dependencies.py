"""
FastAPI dependency injection — provides shared service instances to endpoints.
"""

from functools import lru_cache

from app.ai.llm_client import LLMClient
from app.services.analysis_service import AnalysisService
from app.vectordb.chroma_client import ChromaVectorStore

# ── Singletons ────────────────────────────────────────────────────────

_vector_store: ChromaVectorStore | None = None
_llm_client: LLMClient | None = None
_analysis_service: AnalysisService | None = None


def init_dependencies() -> None:
    """Initialize all shared service instances. Called once at app startup."""
    global _vector_store, _llm_client, _analysis_service

    _vector_store = ChromaVectorStore()
    _vector_store.initialize()

    _llm_client = LLMClient()

    _analysis_service = AnalysisService(
        vector_store=_vector_store,
        llm_client=_llm_client,
    )


def get_vector_store() -> ChromaVectorStore:
    """FastAPI dependency: get the ChromaDB vector store."""
    if _vector_store is None:
        raise RuntimeError("Vector store not initialized. Call init_dependencies() first.")
    return _vector_store


def get_llm_client() -> LLMClient:
    """FastAPI dependency: get the LLM client."""
    if _llm_client is None:
        raise RuntimeError("LLM client not initialized. Call init_dependencies() first.")
    return _llm_client


def get_analysis_service() -> AnalysisService:
    """FastAPI dependency: get the analysis service."""
    if _analysis_service is None:
        raise RuntimeError("Analysis service not initialized. Call init_dependencies() first.")
    return _analysis_service
