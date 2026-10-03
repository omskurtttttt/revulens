"""
Pydantic API Schemas for RevuLens FastAPI Service.

Adheres strictly to GEMINI.md API contract:
- POST /classify -> { label, display_label, confidence }
- POST /explain  -> { tokens: [{ text, weight }], base_value }
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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


class ClassifyResponse(BaseModel):
    """
    Response schema for POST /classify.
    Strictly follows GEMINI.md contract: { label, display_label, confidence }.
    """
    label: str = Field(
        ...,
        description="Internal model class: 'Genuine' or 'Deceptive'."
    )
    display_label: str = Field(
        ...,
        description="User-facing display label: 'Likely Genuine' or 'Potentially Deceptive'."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated model confidence score between 0.0 and 1.0."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Server-side processing time in milliseconds (ISO/IEC 25010 metric)."
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


class TokenAttribution(BaseModel):
    """Individual word/token attribution weight."""
    text: str = Field(..., description="Token text string.")
    weight: float = Field(
        ...,
        description="SHAP attribution weight. Positive (>0) -> Deceptive, Negative (<0) -> Genuine."
    )


class ExplainResponse(BaseModel):
    """
    Response schema for POST /explain.
    Strictly follows GEMINI.md contract: { tokens: [{ text, weight }], base_value }.
    """
    tokens: List[TokenAttribution] = Field(
        ...,
        description="Ordered list of word tokens with signed SHAP contribution weights."
    )
    base_value: float = Field(
        ...,
        description="SHAP base value (expected P(Deceptive))."
    )
    prediction_deceptive_prob: Optional[float] = Field(
        default=None,
        description="Calibrated probability of Deceptive class."
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
