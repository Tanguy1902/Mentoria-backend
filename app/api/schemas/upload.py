"""
Pydantic schemas for the /upload endpoint.
"""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response after a successful file upload."""
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension (.pdf, .pptx)")
    text_length: int = Field(..., description="Number of characters in extracted text")
    page_count: int = Field(..., description="Number of pages/slides")
    uploaded_at: str = Field(..., description="ISO 8601 upload timestamp")
    message: str = Field(default="Document uploaded and text extracted successfully")
