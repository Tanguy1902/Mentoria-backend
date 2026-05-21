"""
Custom exception hierarchy for the application.
Each exception maps to specific HTTP status codes via global handlers.
"""

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


# ── Base ──────────────────────────────────────────────────────────────
class AppError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(self.message)


# ── Document errors ───────────────────────────────────────────────────
class DocumentNotFoundError(AppError):
    """Raised when a requested document does not exist."""
    pass


class UnsupportedFileTypeError(AppError):
    """Raised when an uploaded file type is not supported."""
    pass


class FileTooLargeError(AppError):
    """Raised when an uploaded file exceeds the size limit."""
    pass


# ── Parsing errors ────────────────────────────────────────────────────
class ParsingError(AppError):
    """Raised when text extraction from a document fails."""
    pass


# ── Indexing errors ───────────────────────────────────────────────────
class IndexingError(AppError):
    """Raised when document indexing into the vector store fails."""
    pass


class DocumentNotIndexedError(AppError):
    """Raised when analysis is requested for a non-indexed document."""
    pass


# ── AI / LLM errors ──────────────────────────────────────────────────
class LLMError(AppError):
    """Raised when the LLM API call fails."""
    pass


class AnalysisError(AppError):
    """Raised when the analysis pipeline fails."""
    pass


# ── Global exception handlers ────────────────────────────────────────

def _error_response(status_code: int, error_type: str, message: str, detail: str | None = None) -> JSONResponse:
    content: dict = {
        "error": error_type,
        "message": message,
    }
    if detail:
        content["detail"] = detail
    return JSONResponse(status_code=status_code, content=content)


async def document_not_found_handler(_request: Request, exc: DocumentNotFoundError) -> JSONResponse:
    return _error_response(status.HTTP_404_NOT_FOUND, "document_not_found", exc.message, exc.detail)


async def unsupported_file_type_handler(_request: Request, exc: UnsupportedFileTypeError) -> JSONResponse:
    return _error_response(status.HTTP_400_BAD_REQUEST, "unsupported_file_type", exc.message, exc.detail)


async def file_too_large_handler(_request: Request, exc: FileTooLargeError) -> JSONResponse:
    return _error_response(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file_too_large", exc.message, exc.detail)


async def parsing_error_handler(_request: Request, exc: ParsingError) -> JSONResponse:
    return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "parsing_error", exc.message, exc.detail)


async def indexing_error_handler(_request: Request, exc: IndexingError) -> JSONResponse:
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "indexing_error", exc.message, exc.detail)


async def document_not_indexed_handler(_request: Request, exc: DocumentNotIndexedError) -> JSONResponse:
    return _error_response(status.HTTP_400_BAD_REQUEST, "document_not_indexed", exc.message, exc.detail)


async def llm_error_handler(_request: Request, exc: LLMError) -> JSONResponse:
    return _error_response(status.HTTP_502_BAD_GATEWAY, "llm_error", exc.message, exc.detail)


async def analysis_error_handler(_request: Request, exc: AnalysisError) -> JSONResponse:
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "analysis_error", exc.message, exc.detail)


# Map exception classes to their handlers for registration
EXCEPTION_HANDLERS: dict[type[Exception], object] = {
    DocumentNotFoundError: document_not_found_handler,
    UnsupportedFileTypeError: unsupported_file_type_handler,
    FileTooLargeError: file_too_large_handler,
    ParsingError: parsing_error_handler,
    IndexingError: indexing_error_handler,
    DocumentNotIndexedError: document_not_indexed_handler,
    LLMError: llm_error_handler,
    AnalysisError: analysis_error_handler,
}
