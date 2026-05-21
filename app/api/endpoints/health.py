"""
GET /api/v1/health — Service health check.
"""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_vector_store
from app.api.schemas.health import HealthResponse
from app.config import get_settings
from app.core.logging import get_logger
from app.vectordb.chroma_client import ChromaVectorStore

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and its dependencies.",
)
async def health_check(
    vector_store: ChromaVectorStore = Depends(get_vector_store),
) -> HealthResponse:
    """Return the health status of all service components."""
    settings = get_settings()

    components: dict[str, str] = {}

    # Check ChromaDB
    chroma_ok = vector_store.is_healthy()
    components["chromadb"] = "healthy" if chroma_ok else "unhealthy"

    # Check OpenRouter API key is configured
    api_key_set = bool(settings.OPENROUTER_API_KEY and settings.OPENROUTER_API_KEY != "sk-or-v1-your-key-here")
    components["openrouter_api_key"] = "configured" if api_key_set else "missing"

    # Overall status
    all_healthy = chroma_ok and api_key_set
    overall_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        components=components,
        model=settings.OPENROUTER_MODEL,
    )
