"""
Document service: file upload handling, persistence, and text extraction orchestration.
"""

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import anyio
from fastapi import UploadFile

from app.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    FileTooLargeError,
    ParsingError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.parsers.base import ParsedDocument
from app.parsers.pdf_parser import PDFParser
from app.parsers.ppt_parser import PPTParser

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".ppt"}
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024
DOCUMENTS_DB_NAME = "documents.sqlite3"


@dataclass
class DocumentInfo:
    """Metadata about an uploaded document."""
    document_id: str
    filename: str
    file_type: str
    file_path: str
    text_length: int
    page_count: int
    uploaded_at: str
    text_path: str  # path to extracted text cache


_documents: dict[str, DocumentInfo] = {}
_documents_loaded = False
_documents_lock = threading.RLock()
_parsers = [PDFParser(), PPTParser()]


def _documents_db_path() -> Path:
    """Path to the metadata database used for persisted document records."""
    settings = get_settings()
    return settings.upload_path / DOCUMENTS_DB_NAME


def _open_documents_db() -> sqlite3.Connection:
    """Open the persisted document metadata database and ensure the schema exists."""
    db_path = _documents_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            text_length INTEGER NOT NULL,
            page_count INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL,
            text_path TEXT NOT NULL
        )
        """
    )
    return conn


def _row_to_document_info(row: sqlite3.Row) -> DocumentInfo:
    """Convert a database row into a ``DocumentInfo`` instance."""
    return DocumentInfo(
        document_id=row["document_id"],
        filename=row["filename"],
        file_type=row["file_type"],
        file_path=row["file_path"],
        text_length=int(row["text_length"]),
        page_count=int(row["page_count"]),
        uploaded_at=row["uploaded_at"],
        text_path=row["text_path"],
    )


def _ensure_documents_loaded() -> None:
    """Load persisted document metadata into memory once per process."""
    global _documents_loaded

    if _documents_loaded:
        return

    with _documents_lock:
        if _documents_loaded:
            return

        with _open_documents_db() as conn:
            rows = conn.execute(
                """
                SELECT document_id, filename, file_type, file_path, text_length, page_count,
                       uploaded_at, text_path
                FROM documents
                ORDER BY uploaded_at DESC
                """
            ).fetchall()

        _documents.clear()
        for row in rows:
            doc_info = _row_to_document_info(row)
            _documents[doc_info.document_id] = doc_info

        _documents_loaded = True
        logger.info("Loaded %d persisted document metadata record(s)", len(_documents))


def _persist_document(doc_info: DocumentInfo) -> None:
    """Upsert a document metadata record in the persistence layer."""
    with _open_documents_db() as conn:
        conn.execute(
            """
            INSERT INTO documents (
                document_id, filename, file_type, file_path, text_length, page_count,
                uploaded_at, text_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                filename=excluded.filename,
                file_type=excluded.file_type,
                file_path=excluded.file_path,
                text_length=excluded.text_length,
                page_count=excluded.page_count,
                uploaded_at=excluded.uploaded_at,
                text_path=excluded.text_path
            """,
            (
                doc_info.document_id,
                doc_info.filename,
                doc_info.file_type,
                doc_info.file_path,
                doc_info.text_length,
                doc_info.page_count,
                doc_info.uploaded_at,
                doc_info.text_path,
            ),
        )
        conn.commit()


def _get_parser(extension: str):
    """Find the appropriate parser for a file extension."""
    for parser in _parsers:
        if parser.supports(extension):
            return parser
    return None


async def save_upload(file: UploadFile) -> DocumentInfo:
    """Save an uploaded file to disk and extract its text.

    Args:
        file: The uploaded file from FastAPI.

    Returns:
        A ``DocumentInfo`` with all metadata about the processed file.

    Raises:
        UnsupportedFileTypeError: If the file type is not PDF or PPTX.
        FileTooLargeError: If the file exceeds the size limit.
        ParsingError: If text extraction fails.
    """
    settings = get_settings()
    filename = file.filename or "unknown"
    extension = Path(filename).suffix.lower()

    # Validate file type
    if extension not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: '{extension}'",
            detail=f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Generate unique document ID
    document_id = str(uuid.uuid4()).replace("-", "")[:16]
    logger.info("Processing upload: %s (id=%s)", filename, document_id)

    # Save file to disk
    upload_dir = settings.upload_path / document_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename

    written_bytes = 0
    try:
        async with aiofiles.open(file_path, "wb") as output_file:
            while True:
                chunk = await file.read(UPLOAD_READ_CHUNK_SIZE)
                if not chunk:
                    break

                written_bytes += len(chunk)
                if written_bytes > settings.max_file_size_bytes:
                    raise FileTooLargeError(
                        f"File too large: {written_bytes / (1024*1024):.1f}MB",
                        detail=f"Maximum allowed size: {settings.MAX_FILE_SIZE_MB}MB",
                    )

                await output_file.write(chunk)

        logger.info("File saved: %s (%d bytes)", file_path, written_bytes)
    except (FileTooLargeError, UnsupportedFileTypeError):
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise ParsingError(
            f"Failed to save uploaded file: {filename}",
            detail=str(exc),
        ) from exc
    finally:
        await file.close()

    # Extract text
    parser = _get_parser(extension)
    if parser is None:
        raise UnsupportedFileTypeError(f"No parser available for: {extension}")

    text_path = upload_dir / "extracted_text.txt"
    try:
        parsed: ParsedDocument = await anyio.to_thread.run_sync(parser.parse, file_path)

        # Cache extracted text
        async with aiofiles.open(text_path, "w", encoding="utf-8") as text_file:
            await text_file.write(parsed.text)
    except Exception as exc:
        if text_path.exists():
            text_path.unlink(missing_ok=True)
        if file_path.exists():
            file_path.unlink(missing_ok=True)
        raise ParsingError(
            f"Failed to parse uploaded file: {filename}",
            detail=str(exc),
        ) from exc

    # Store document info
    doc_info = DocumentInfo(
        document_id=document_id,
        filename=filename,
        file_type=extension,
        file_path=str(file_path),
        text_length=len(parsed.text),
        page_count=parsed.page_count,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        text_path=str(text_path),
    )
    _ensure_documents_loaded()
    _persist_document(doc_info)
    with _documents_lock:
        _documents[document_id] = doc_info

    logger.info(
        "Upload complete: id=%s, pages=%d, chars=%d",
        document_id,
        parsed.page_count,
        len(parsed.text),
    )
    return doc_info


def get_document(document_id: str) -> DocumentInfo:
    """Retrieve document info by ID.

    Raises:
        DocumentNotFoundError: If the document ID is not found.
    """
    _ensure_documents_loaded()

    cached = _documents.get(document_id)
    if cached is not None:
        return cached

    with _open_documents_db() as conn:
        row = conn.execute(
            """
            SELECT document_id, filename, file_type, file_path, text_length, page_count,
                   uploaded_at, text_path
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

    if row is None:
        raise DocumentNotFoundError(
            f"Document not found: {document_id}",
            detail="Upload a document first using /api/v1/upload",
        )

    doc_info = _row_to_document_info(row)
    with _documents_lock:
        _documents[document_id] = doc_info
    return doc_info


def get_document_text(document_id: str) -> str:
    """Read the extracted text for a document.

    Raises:
        DocumentNotFoundError: If the document ID is not found.
        ParsingError: If the cached text file cannot be read.
    """
    doc = get_document(document_id)
    text_path = Path(doc.text_path)

    if not text_path.exists():
        raise ParsingError(
            f"Extracted text file missing for document: {document_id}",
            detail=f"Expected at: {text_path}",
        )

    return text_path.read_text(encoding="utf-8")


def list_documents() -> list[DocumentInfo]:
    """Return all uploaded documents."""
    _ensure_documents_loaded()
    return list(_documents.values())
