"""
POST /api/v1/index — Chunk and index a document into the vector store.
"""

import time

import anyio
from fastapi import APIRouter, Depends

from app.api.dependencies import get_vector_store
from app.api.schemas.index import IndexRequest, IndexResponse
from app.core.logging import get_logger
from app.services import document_service
from app.services.chunking_service import chunk_text
from app.vectordb.chroma_client import ChromaVectorStore

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Index a document",
    description="Chunk the uploaded document's text and store embeddings "
                "in the vector database for RAG retrieval.",
)
async def index_document(
    request: IndexRequest,
    vector_store: ChromaVectorStore = Depends(get_vector_store),
) -> IndexResponse:
    """Chunk document text, generate embeddings, and store in ChromaDB."""
    logger.info("Index request for document: %s", request.document_id)
    started_at = time.perf_counter()

    # Get the extracted text
    text = await anyio.to_thread.run_sync(document_service.get_document_text, request.document_id)

    # Chunk the text
    chunks = await anyio.to_thread.run_sync(
        chunk_text,
        text,
        request.document_id,
        request.chunk_size,
        request.chunk_overlap,
    )

    if not chunks:
        return IndexResponse(
            document_id=request.document_id,
            chunks_count=0,
            status="empty",
            message="No chunks were created — the document text may be too short.",
        )

    # Store chunks in vector database
    collection_name = f"doc_{request.document_id}"
    count = await anyio.to_thread.run_sync(
        vector_store.add_chunks,
        collection_name,
        [c.chunk_id for c in chunks],
        [c.text for c in chunks],
        [c.metadata for c in chunks],
    )

    logger.info("Indexed %d chunks for document %s", count, request.document_id)
    logger.info("Indexing pipeline finished in %.2fs", time.perf_counter() - started_at)

    return IndexResponse(
        document_id=request.document_id,
        chunks_count=count,
        status="indexed",
        message=f"Document indexed successfully with {count} chunks",
    )
