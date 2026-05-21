"""
Pydantic schemas for the /health endpoint.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status: healthy | degraded | unhealthy")
    version: str = Field(..., description="Application version")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Status of each sub-component",
    )
    model: str = Field(..., description="Configured LLM model name")
