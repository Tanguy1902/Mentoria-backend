"""
POST /api/v1/analyze — Run RAG-powered academic analysis on a document.
"""

from dataclasses import asdict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_analysis_service
from app.api.schemas.analyze import AnalyzeRequest, AnalyzeResponse, ChatRequest
from app.core.logging import get_logger
from app.services.analysis_service import AnalysisService

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    summary="Chat with document",
    description="Interactive academic jury chat with the document using RAG.",
)
async def chat_with_document(
    request: ChatRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service),
):
    """Stream chat responses."""
    logger.info("Chat request for document: %s", request.document_id)
    
    return StreamingResponse(
        analysis_service.chat_with_document(
            document_id=request.document_id,
            message=request.message,
            chat_history=[h.model_dump() for h in request.history]
        ),
        media_type="text/event-stream"
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a document",
    description="Perform a full academic analysis using RAG: retrieve relevant "
                "chunks, generate jury questions, critical remarks, improvement "
                "suggestions, and an optional scoring rubric.",
)
async def analyze_document(
    request: AnalyzeRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service),
) -> AnalyzeResponse:
    """Run the full RAG analysis pipeline."""
    logger.info(
        "Analysis request for document: %s (custom_query=%s, rubric=%s)",
        request.document_id,
        bool(request.custom_query),
        request.include_rubric,
    )

    result = await analysis_service.analyze_document(
        document_id=request.document_id,
        custom_query=request.custom_query,
        include_rubric=request.include_rubric,
    )

    # Convert dataclass to dict for Pydantic response
    result_dict = asdict(result)

    return AnalyzeResponse(**result_dict)
