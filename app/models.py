from pydantic import BaseModel, Field, field_validator
from typing import Optional


class AssessmentRequest(BaseModel):
    patient_id: str = Field(
        min_length=1,
        max_length=20,
        pattern=r'^PT-\d{3}$',
        description="Patient ID in PT-XXX format",
    )


class Citation(BaseModel):
    source: str = "NG12 PDF"
    page: int = Field(ge=0, le=500)
    chunk_id: str
    excerpt: str


class RiskAssessmentResponse(BaseModel):
    patient_id: str
    patient_name: str
    risk_level: str
    cancer_type_suspected: Optional[str] = None
    reasoning: str
    recommendations: list[str]
    citations: list[Citation]
    disclaimer: Optional[str] = None
    grounding_flags: Optional[list[str]] = None


class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Session identifier (alphanumeric, hyphens, underscores)",
    )
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="Chat message (1-2000 characters)",
    )
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message must not be blank or whitespace-only")
        return v


class ChatStreamRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Session identifier (alphanumeric, hyphens, underscores)",
    )
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="Chat message (1-2000 characters)",
    )
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message must not be blank or whitespace-only")
        return v


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation]
    disclaimer: Optional[str] = None
    grounding_flags: Optional[list[str]] = None


# ── LLM Judge Models ───────────────────────────────────────────────────

class CriterionResult(BaseModel):
    """Result of a single evaluation criterion (binary PASS/FAIL with CoT reasoning)."""
    verdict: str = Field(description="PASS or FAIL")
    reasoning: str = Field(description="Chain-of-thought reasoning for the verdict")


class JudgeVerdict(BaseModel):
    """Multi-criteria evaluation result from the LLM judge."""
    overall_verdict: str = Field(description="PASS or FAIL — FAIL if any critical criterion fails")
    score: str = Field(description="e.g. '4/5' — count of passed criteria")
    criteria: dict[str, CriterionResult] = Field(description="Per-criterion verdicts")
    critical_issues: list[str] = Field(default_factory=list)
    cross_examination: Optional[list[dict]] = Field(
        default=None,
        description="Cross-examination results for faithfulness verification",
    )
