"""
Pydantic schemas for the /analyze endpoint.
"""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Request body to analyze a document."""
    document_id: str = Field(..., description="ID of the document to analyze")
    custom_query: str | None = Field(
        default=None,
        description="Optional specific area to focus the analysis on",
    )
    include_rubric: bool = Field(
        default=True,
        description="Whether to include the scoring rubric in the response",
    )


class JuryQuestionSchema(BaseModel):
    question: str
    rationale: str
    section_reference: str


class CriticalRemarkSchema(BaseModel):
    remark: str
    severity: str
    suggestion: str


class ImprovementSuggestionSchema(BaseModel):
    area: str
    current_state: str
    recommended_action: str


class RubricScoreSchema(BaseModel):
    score: int
    max: int
    comment: str


class ScoringRubricSchema(BaseModel):
    methodology: RubricScoreSchema
    clarity: RubricScoreSchema
    technical_correctness: RubricScoreSchema
    argumentation: RubricScoreSchema
    originality: RubricScoreSchema
    overall: RubricScoreSchema


class AnalyzeResponse(BaseModel):
    """Full structured analysis response."""
    document_id: str
    document_title: str
    jury_questions: list[JuryQuestionSchema] = Field(default_factory=list)
    critical_remarks: list[CriticalRemarkSchema] = Field(default_factory=list)
    improvement_suggestions: list[ImprovementSuggestionSchema] = Field(default_factory=list)
    scoring_rubric: ScoringRubricSchema | None = None
    model_used: str = ""
    chunks_used: int = 0
    message: str = Field(default="Analysis completed successfully")


class ChatHistoryItem(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    document_id: str
    message: str
    history: list[ChatHistoryItem] = Field(default_factory=list)
