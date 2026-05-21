"""
API router aggregator — registers all endpoint routers under /api/v1.
"""

from fastapi import APIRouter

from app.api.endpoints import analyze, health, index, upload, reference

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(
    upload.router,
    tags=["Upload"],
)

api_router.include_router(
    index.router,
    tags=["Indexing"],
)

api_router.include_router(
    analyze.router,
    tags=["Analysis"],
)

api_router.include_router(
    reference.router,
    prefix="/reference-questions",
    tags=["Reference Data"],
)

api_router.include_router(
    health.router,
    tags=["Health"],
)
