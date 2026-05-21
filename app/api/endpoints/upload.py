"""
POST /api/v1/upload — Upload a PDF or PowerPoint document.
"""

from fastapi import APIRouter, File, UploadFile

from app.api.schemas.upload import UploadResponse
from app.core.logging import get_logger
from app.services import document_service

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document",
    description="Upload a PDF or PowerPoint file for analysis. "
                "The file will be saved and its text content extracted automatically.",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or PPTX file to upload"),
) -> UploadResponse:
    """Handle document upload, save to disk, and extract text."""
    logger.info("Upload request received: %s", file.filename)

    doc_info = await document_service.save_upload(file)

    return UploadResponse(
        document_id=doc_info.document_id,
        filename=doc_info.filename,
        file_type=doc_info.file_type,
        text_length=doc_info.text_length,
        page_count=doc_info.page_count,
        uploaded_at=doc_info.uploaded_at,
    )
