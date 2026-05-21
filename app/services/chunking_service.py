"""
Chunking service: text normalization, token-aware splitting with overlap.
"""

import re
import uuid
from dataclasses import dataclass

import tiktoken

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Use cl100k_base tokenizer (GPT-4 / modern models)
_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    chunk_id: str
    text: str
    token_count: int
    chunk_index: int
    metadata: dict


def _count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(_encoding.encode(text))


def _normalize_text(text: str) -> str:
    """Clean and normalize text for chunking.

    - Removes excessive whitespace
    - Strips control characters (except newlines)
    - Normalizes line endings
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control characters except newlines and tabs
    text = re.sub(r"[^\S\n\t]+", " ", text)
    # Collapse multiple blank lines into two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments for boundary-aware chunking."""
    # Split on sentence-ending punctuation followed by whitespace
    segments = re.split(r"(?<=[.!?])\s+", text)
    # Also split on double newlines (paragraph boundaries)
    result: list[str] = []
    for segment in segments:
        parts = segment.split("\n\n")
        result.extend(parts)
    return [s.strip() for s in result if s.strip()]


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """Split text into overlapping chunks with sentence-boundary awareness.

    Args:
        text: The full document text to chunk.
        document_id: ID of the source document (for metadata).
        chunk_size: Target tokens per chunk (default from settings).
        chunk_overlap: Overlap tokens between chunks (default from settings).

    Returns:
        A list of ``TextChunk`` objects ready for embedding.
    """
    settings = get_settings()
    target_size = chunk_size or settings.CHUNK_SIZE
    overlap = chunk_overlap or settings.CHUNK_OVERLAP

    logger.info(
        "Chunking document %s: target_size=%d, overlap=%d",
        document_id,
        target_size,
        overlap,
    )

    # Normalize the text
    normalized = _normalize_text(text)
    total_tokens = _count_tokens(normalized)
    logger.info("Normalized text: %d characters, ~%d tokens", len(normalized), total_tokens)

    # Split into sentences
    sentences = _split_into_sentences(normalized)

    if not sentences:
        logger.warning("No sentences found after splitting for document %s", document_id)
        return []

    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_token_count = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        # If a single sentence exceeds chunk size, add it as its own chunk
        if sentence_tokens > target_size:
            # Flush current buffer first
            if current_sentences:
                chunk_text_content = " ".join(current_sentences)
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        text=chunk_text_content,
                        token_count=_count_tokens(chunk_text_content),
                        chunk_index=chunk_index,
                        metadata={
                            "document_id": document_id,
                            "chunk_index": chunk_index,
                        },
                    )
                )
                chunk_index += 1
                current_sentences = []
                current_token_count = 0

            # Add the long sentence as its own chunk
            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    text=sentence,
                    token_count=sentence_tokens,
                    chunk_index=chunk_index,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1
            continue

        # Check if adding this sentence would exceed the target chunk size
        if current_token_count + sentence_tokens > target_size and current_sentences:
            # Create chunk from current buffer
            chunk_text_content = " ".join(current_sentences)
            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    text=chunk_text_content,
                    token_count=_count_tokens(chunk_text_content),
                    chunk_index=chunk_index,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1

            # Overlap: keep the last few sentences for context continuity
            overlap_sentences: list[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                s_tokens = _count_tokens(s)
                if overlap_count + s_tokens > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_count += s_tokens

            current_sentences = overlap_sentences
            current_token_count = overlap_count

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Flush remaining sentences
    if current_sentences:
        chunk_text_content = " ".join(current_sentences)
        chunks.append(
            TextChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                text=chunk_text_content,
                token_count=_count_tokens(chunk_text_content),
                chunk_index=chunk_index,
                metadata={
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                },
            )
        )

    logger.info(
        "Chunking complete: %d chunks (avg ~%d tokens/chunk)",
        len(chunks),
        total_tokens // max(len(chunks), 1),
    )

    return chunks
