"""
Document service: file upload handling, persistence, and text extraction orchestration.
"""

import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

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


# In-memory document registry (production would use a database)
_documents: dict[str, DocumentInfo] = {}
_parsers = [PDFParser(), PPTParser()]


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

    try:
        # Read content and check size
        content = await file.read()
        if len(content) > settings.max_file_size_bytes:
            raise FileTooLargeError(
                f"File too large: {len(content) / (1024*1024):.1f}MB",
                detail=f"Maximum allowed size: {settings.MAX_FILE_SIZE_MB}MB",
            )

        with open(file_path, "wb") as f:
            f.write(content)
        logger.info("File saved: %s (%d bytes)", file_path, len(content))
    except (FileTooLargeError, UnsupportedFileTypeError):
        raise
    except Exception as exc:
        raise ParsingError(
            f"Failed to save uploaded file: {filename}",
            detail=str(exc),
        ) from exc

    # Extract text
    parser = _get_parser(extension)
    if parser is None:
        raise UnsupportedFileTypeError(f"No parser available for: {extension}")

    parsed: ParsedDocument = parser.parse(file_path)

    # Cache extracted text
    text_path = upload_dir / "extracted_text.txt"
    text_path.write_text(parsed.text, encoding="utf-8")

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
    if document_id not in _documents:
        raise DocumentNotFoundError(
            f"Document not found: {document_id}",
            detail="Upload a document first using /api/v1/upload",
        )
    return _documents[document_id]


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
    return list(_documents.values())
