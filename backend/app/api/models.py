"""
Pydantic API Schemas for RevuLens FastAPI Service.

Adheres strictly to GEMINI.md API contract:
- POST /classify -> { label, display_label, truncated, latency_ms }
- POST /explain  -> { tokens: [{ text, weight }], truncated, latency_ms, cached }
- No confidence score, probability, base value, or raw decision score is ever returned.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------
# Classification Schemas
# ---------------------------------------------------------

class ClassifyRequest(BaseModel):
    """Input payload for text review classification."""
    text: str = Field(
        ...,
        min_length=1,
        description="Raw highlighted text selection from the shopping page."
    )

    @field_validator("text")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Review text cannot be blank or whitespace-only.")
        return value


class ClassifyResponse(BaseModel):
    """
    Response schema for POST /classify.
    Strictly follows GEMINI.md contract: { label, display_label, truncated, latency_ms }.
    Never exposes confidence scores, probabilities, or decision function scores.
    """
    label: str = Field(
        ...,
        description="Internal model class: 'Genuine' or 'Deceptive'."
    )
    display_label: str = Field(
        ...,
        description="User-facing display label: 'Likely Genuine' or 'Potentially Deceptive'."
    )
    truncated: bool = Field(
        default=False,
        description="Whether input text exceeded MAX_WORDS and was truncated."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Server-side processing time in milliseconds."
    )


# ---------------------------------------------------------
# Explainability Schemas
# ---------------------------------------------------------

class ExplainRequest(BaseModel):
    """Input payload for POST /explain."""
    text: str = Field(
        ...,
        min_length=1,
        description="Raw highlighted text selection for token-level SHAP explanation."
    )
    max_evals: Optional[int] = Field(
        default=None,
        ge=10,
        le=500,
        description="Maximum model evaluations for SHAP (default: configured in settings)."
    )
    max_words: Optional[int] = Field(
        default=None,
        ge=5,
        le=300,
        description="Maximum word count to evaluate before truncation."
    )

    @field_validator("text")
    @classmethod
    def validate_non_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Review text cannot be blank or whitespace-only.")
        return value


class TokenAttribution(BaseModel):
    """Individual word/token attribution weight."""
    text: str = Field(..., description="Token text string.")
    weight: float = Field(
        ...,
        description="SHAP attribution weight. Positive (>0) pushes toward Deceptive, Negative (<0) toward Genuine."
    )


class ExplainResponse(BaseModel):
    """
    Response schema for POST /explain.
    Strictly follows GEMINI.md contract: { tokens: [{ text, weight }], truncated, latency_ms, cached }.
    Never exposes confidence scores, probabilities, base value, or decision function scores.
    """
    tokens: List[TokenAttribution] = Field(
        ...,
        description="Ordered list of word tokens with signed SHAP contribution weights."
    )
    truncated: bool = Field(
        default=False,
        description="Whether input text exceeded MAX_WORDS and was truncated."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Server-side explanation response time in milliseconds."
    )
    cached: Optional[bool] = Field(
        default=False,
        description="Indicates whether the explanation was retrieved from in-memory cache."
    )


# ---------------------------------------------------------
# Health Check Schema
# ---------------------------------------------------------

class HealthResponse(BaseModel):
    """System health check and diagnostic metadata."""
    status: str = Field(default="ok", description="Service health status.")
    model_loaded: bool = Field(..., description="Whether the hybrid model pipeline is loaded.")
    pooling_strategy: str = Field(default="mean", description="DistilBERT pooling strategy.")
    device: str = Field(..., description="Execution compute device ('cpu' or 'cuda').")
    version: str = Field(default="1.0.0", description="API version.")
