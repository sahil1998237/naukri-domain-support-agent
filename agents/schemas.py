"""
Pydantic Schemas for Naukri.com Domain Support Agent (Task 9).

Implements strict structured output validation for all agent and API responses.
Guarantees consistent schema contracts across:
- CrewAI multi-agent synthesis
- Memory layer state tracking
- FastAPI HTTP / WebSocket endpoints (Day 8)
- AutoGen review stage (Day 11)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PolicyCitation(BaseModel):
    """Structured citation metadata for grounded policy retrieval."""
    doc_id: str = Field(..., description="Unique policy document identifier (e.g. POL-003)")
    title: str = Field(..., description="Formal policy title")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score [0.0, 1.0]")
    excerpt: Optional[str] = Field(None, description="Relevant sentence or clause used in the response")


class ApplicationStatusInfo(BaseModel):
    """Structured details for candidate job application status lookups."""
    record_id: str = Field(..., description="Unique application identifier (e.g. APP-003)")
    category: str = Field(..., description="Job role category")
    status: str = Field(..., description="Current pipeline stage")
    days_since_created: int = Field(..., ge=0, description="Elapsed days since application was submitted")
    flagged_priority_review: bool = Field(..., description="Whether recruiter flagged application for priority")
    escalation_score: float = Field(..., ge=0.0, le=1.0, description="Continuous escalation score [0.0, 1.0]")
    escalation_threshold: float = Field(..., ge=0.0, le=1.0, description="Empirical 80th percentile threshold")
    escalation_recommended: bool = Field(..., description="True if escalation_score >= escalation_threshold")


class NaukriSupportResponse(BaseModel):
    """
    Standardized, validated output contract for the Naukri.com Domain Support Agent.
    Every response returned to candidates, APIs, or reviewers adheres to this model.
    """
    query: str = Field(..., min_length=1, description="Original user prompt or query")
    answer: str = Field(..., min_length=1, description="Synthesized, grounded answer text")
    sources: List[str] = Field(default_factory=list, description="List of source policy IDs or database records cited")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Model confidence score [0.0, 1.0]")
    escalation_needed: bool = Field(default=False, description="Whether human managerial intervention is required")
    record_id: Optional[str] = Field(None, description="Application ID if detected or referenced")
    application_details: Optional[ApplicationStatusInfo] = Field(None, description="Detailed application status if lookup occurred")
    citations: List[PolicyCitation] = Field(default_factory=list, description="Structured policy citations if RAG retrieval occurred")
    agents_involved: List[str] = Field(default_factory=list, description="Roles of agents that participated in generating this response")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversational tracking")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of response generation",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operational metadata (tokens, latency, etc.)")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence score {v} must be between 0.0 and 1.0")
        return round(v, 4)

    @field_validator("answer")
    @classmethod
    def validate_answer_not_blank(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Answer cannot be empty or whitespace only")
        return clean
