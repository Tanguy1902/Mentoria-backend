"""
Embedding generation service using ChromaDB's built-in embedding function.
Uses all-MiniLM-L6-v2 from sentence-transformers (runs locally).
"""

from chromadb.utils import embedding_functions

from app.core.logging import get_logger

logger = get_logger(__name__)

# ChromaDB default: all-MiniLM-L6-v2 (384 dimensions, runs locally)
_embedding_fn: embedding_functions.DefaultEmbeddingFunction | None = None


def get_embedding_function() -> embedding_functions.DefaultEmbeddingFunction:
    """Get or create the singleton embedding function.

    Returns:
        A ChromaDB-compatible embedding function instance.
    """
    global _embedding_fn
    if _embedding_fn is None:
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        _embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        logger.info("Embedding model loaded successfully")
    return _embedding_fn


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.
    """
    logger.debug("Generating embeddings for %d texts", len(texts))
    fn = get_embedding_function()
    embeddings = fn(texts)
    logger.debug("Generated %d embeddings", len(embeddings))
    return embeddings
