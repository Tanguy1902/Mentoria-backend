"""
Pydantic schemas for the /index endpoint.
"""

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    """Request body to index a document."""
    document_id: str = Field(..., description="ID of the document to index")
    chunk_size: int | None = Field(default=None, description="Optional custom chunk size (tokens)")
    chunk_overlap: int | None = Field(default=None, description="Optional custom chunk overlap (tokens)")


class IndexResponse(BaseModel):
    """Response after successful document indexing."""
    document_id: str = Field(..., description="ID of the indexed document")
    chunks_count: int = Field(..., description="Number of chunks created and indexed")
    status: str = Field(default="indexed", description="Indexing status")
    message: str = Field(default="Document indexed successfully")
