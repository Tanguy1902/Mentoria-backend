"""
ChromaDB persistent client wrapper for vector storage and retrieval.
"""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.core.exceptions import IndexingError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store with its relevance score."""
    chunk_id: str
    text: str
    metadata: dict
    distance: float


class ChromaVectorStore:
    """Wrapper around ChromaDB for document chunk storage and retrieval."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client: chromadb.ClientAPI | None = None
        self._persist_dir = str(settings.chroma_path)

    def initialize(self) -> None:
        """Create the persistent ChromaDB client. Call once at startup."""
        logger.info("Initializing ChromaDB at: %s", self._persist_dir)
        try:
            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info("ChromaDB initialized successfully")
        except Exception as exc:
            logger.error("Failed to initialize ChromaDB: %s", exc)
            raise IndexingError(
                "Failed to initialize vector database",
                detail=str(exc),
            ) from exc

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            raise IndexingError("ChromaDB client not initialized. Call initialize() first.")
        return self._client

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a collection with cosine similarity.

        Args:
            name: Collection name (typically the document ID).

        Returns:
            A ChromaDB collection.
        """
        logger.debug("Getting/creating collection: %s", name)
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.get_collection(name=name)
            return True
        except Exception:
            return False

    def add_chunks(
        self,
        collection_name: str,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> int:
        """Add document chunks to a collection in batch.

        Args:
            collection_name: Target collection name.
            chunk_ids: Unique IDs for each chunk.
            documents: Text content of each chunk.
            metadatas: Metadata dicts for each chunk.

        Returns:
            Number of chunks added.

        Raises:
            IndexingError: If the batch upsert fails.
        """
        logger.info(
            "Adding %d chunks to collection '%s'",
            len(documents),
            collection_name,
        )

        try:
            collection = self.get_or_create_collection(collection_name)
            # Batch upsert (ChromaDB handles embedding generation internally)
            collection.upsert(
                ids=chunk_ids,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Successfully indexed %d chunks", len(documents))
            return len(documents)
        except Exception as exc:
            logger.error("Failed to index chunks: %s", exc)
            raise IndexingError(
                f"Failed to index chunks into collection '{collection_name}'",
                detail=str(exc),
            ) from exc

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
    ) -> list[RetrievedChunk]:
        """Query the vector store for relevant chunks.

        Args:
            collection_name: Collection to search.
            query_text: The search query.
            n_results: Maximum number of results to return.

        Returns:
            A list of ``RetrievedChunk`` objects sorted by relevance.
        """
        logger.debug(
            "Querying collection '%s' with: %.80s...",
            collection_name,
            query_text,
        )

        try:
            collection = self.client.get_collection(name=collection_name)
            results = collection.query(
                query_texts=[query_text],
                n_results=min(n_results, collection.count()),
            )
        except Exception as exc:
            logger.error("Vector search failed: %s", exc)
            raise IndexingError(
                f"Failed to query collection '{collection_name}'",
                detail=str(exc),
            ) from exc

        chunks: list[RetrievedChunk] = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        distance=results["distances"][0][i] if results["distances"] else 0.0,
                    )
                )

        logger.info("Retrieved %d chunks from '%s'", len(chunks), collection_name)
        return chunks

    def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data."""
        try:
            self.client.delete_collection(name=name)
            logger.info("Deleted collection: %s", name)
        except Exception as exc:
            logger.warning("Failed to delete collection '%s': %s", name, exc)

    def is_healthy(self) -> bool:
        """Check that the ChromaDB client is responsive."""
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False
