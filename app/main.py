"""
FastAPI application factory with lifespan management.
Entry point: uvicorn app.main:app
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import init_dependencies
from app.api.router import api_router
from app.config import get_settings
from app.core.exceptions import EXCEPTION_HANDLERS
from app.core.logging import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup & shutdown logic."""
    # ── Startup ───────────────────────────────────────────────────────
    setup_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("Model: %s", settings.OPENROUTER_MODEL)
    logger.info("ChromaDB: %s", settings.CHROMA_PERSIST_DIR)
    logger.info("Upload dir: %s", settings.UPLOAD_DIR)
    logger.info("Chunk size: %d tokens, overlap: %d tokens", settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    logger.info("=" * 60)

    # Initialize services
    init_dependencies()
    logger.info("All dependencies initialized successfully")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Shutting down %s", settings.APP_NAME)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "MentorIA — API de production pour l'analyse de documents académiques "
            "(PDF / PowerPoint). Génère des questions de jury, des remarques critiques, "
            "des suggestions d'amélioration et des grilles de notation via RAG + LLM."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────
    for exc_class, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exc_class, handler)

    # ── Routers ───────────────────────────────────────────────────────
    app.include_router(api_router)

    return app


# Application instance
app = create_app()
