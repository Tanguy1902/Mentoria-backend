"""
POST /api/v1/index — Chunk and index a document into the vector store.
"""

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

    # Get the extracted text
    text = document_service.get_document_text(request.document_id)

    # Chunk the text
    chunks = chunk_text(
        text=text,
        document_id=request.document_id,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
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
    count = vector_store.add_chunks(
        collection_name=collection_name,
        chunk_ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )

    logger.info("Indexed %d chunks for document %s", count, request.document_id)

    return IndexResponse(
        document_id=request.document_id,
        chunks_count=count,
        status="indexed",
        message=f"Document indexed successfully with {count} chunks",
    )
