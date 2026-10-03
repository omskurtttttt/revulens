"""
RevuLens FastAPI Application (Step 7).

Adheres strictly to GEMINI.md:
1. POST /classify -> { label, display_label, confidence }
2. POST /explain  -> { tokens: [{ text, weight }], base_value }
3. Both endpoints utilize the shared preprocessing function (backend.app.preprocessing.normalize_text).
4. Non-blocking architecture: /classify is separate and never blocked by /explain.
5. Configurable CORS origins and backend settings (supports chrome-extension:// origins).
6. ISO/IEC 25010 logging: logs response times for /classify and /explain separately.
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.config import settings
from backend.app.api.models import (
    ClassifyRequest,
    ClassifyResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
)
from backend.app.services.embedding import DistilBERTEmbeddingExtractor
from backend.app.services.classifier import ClassificationService
from backend.app.services.explainer import SHAPExplainerService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RevuLens] %(message)s"
)
logger = logging.getLogger("revulens")


# ---------------------------------------------------------
# Application Lifespan & Service Singleton Management
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts and initialize services on application startup.
    Shares a single frozen DistilBERT embedding extractor between
    ClassificationService and SHAPExplainerService to conserve RAM.
    """
    logger.info("Initializing RevuLens ML Pipeline...")
    start_load = time.perf_counter()

    # 1. Initialize shared frozen DistilBERT extractor
    extractor = DistilBERTEmbeddingExtractor(
        model_name=settings.DISTILBERT_CHECKPOINT,
        pooling_strategy="mean",
        max_length=128
    )
    logger.info(f"Loaded frozen DistilBERT ({settings.DISTILBERT_CHECKPOINT}) on {extractor.device.upper()}")

    # 2. Initialize fast classification service
    classifier_service = ClassificationService(
        model_path=settings.MODEL_PATH,
        extractor=extractor
    )
    logger.info(f"Loaded Calibrated LinearSVC pipeline from {settings.MODEL_PATH}")

    # 3. Initialize SHAP explainer service
    explainer_service = SHAPExplainerService(
        model_path=settings.MODEL_PATH,
        extractor=extractor,
        default_max_evals=settings.DEFAULT_MAX_EVALS,
        max_words=settings.MAX_WORDS,
        cache_size=settings.CACHE_SIZE
    )
    logger.info(f"Loaded SHAPExplainerService (default_max_evals={settings.DEFAULT_MAX_EVALS}, max_words={settings.MAX_WORDS})")

    # Store in app state
    app.state.extractor = extractor
    app.state.classifier_service = classifier_service
    app.state.explainer_service = explainer_service

    load_time = time.perf_counter() - start_load
    logger.info(f"RevuLens backend successfully initialized in {load_time:.2f}s!")

    yield

    logger.info("Shutting down RevuLens backend services...")


# ---------------------------------------------------------
# ISO/IEC 25010 Latency Logging Middleware
# ---------------------------------------------------------

class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware logging response times separately for /classify and /explain per GEMINI.md."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        path = request.url.path
        if path in ("/classify", "/explain"):
            logger.info(
                f"[ISO/IEC 25010] Endpoint: {path} | Status: {response.status_code} | Response Time: {duration_ms:.2f} ms"
            )
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        return response


# ---------------------------------------------------------
# FastAPI App Initialization & CORS
# ---------------------------------------------------------

app = FastAPI(
    title="RevuLens Explainable Review Verification API",
    description="Backend API detecting deceptive e-commerce reviews using a DistilBERT-SVM hybrid pipeline with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan
)

# Add latency logging middleware
app.add_middleware(LatencyLoggingMiddleware)

# Add CORS Middleware supporting browser extensions & local frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/", tags=["General"])
async def root() -> Dict[str, Any]:
    """Root metadata endpoint."""
    return {
        "service": "RevuLens API",
        "description": "DistilBERT-SVM Fake Review Detector with SHAP Explanations",
        "endpoints": {
            "health": "/health",
            "classify": "POST /classify",
            "explain": "POST /explain",
            "docs": "/docs"
        },
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check(request: Request) -> HealthResponse:
    """System health check and diagnostic inspection."""
    extractor = getattr(request.app.state, "extractor", None)
    classifier = getattr(request.app.state, "classifier_service", None)

    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        pooling_strategy=classifier.pooling_strategy if classifier else "mean",
        device=extractor.device if extractor else "cpu",
        version="1.0.0"
    )


@app.post("/classify", response_model=ClassifyResponse, tags=["Inference"])
async def classify_review(request: Request, payload: ClassifyRequest) -> ClassifyResponse:
    """
    Classify a review selection into Likely Genuine or Potentially Deceptive.

    Strictly fulfills GEMINI.md contract: { label, display_label, confidence }.
    Ultra-fast execution (<150 ms) without computing SHAP.
    """
    classifier: ClassificationService = getattr(request.app.state, "classifier_service", None)
    if not classifier:
        raise HTTPException(status_code=503, detail="Classification model not ready.")

    try:
        result = classifier.classify(payload.text)
        return ClassifyResponse(
            label=result["label"],
            display_label=result["display_label"],
            confidence=result["confidence"],
            latency_ms=result.get("latency_ms")
        )
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_review(request: Request, payload: ExplainRequest) -> ExplainResponse:
    """
    Compute word-level SHAP attributions for a review selection.

    Strictly fulfills GEMINI.md contract: { tokens: [{ text, weight }], base_value }.
    Runs asynchronously and utilizes in-memory LRU caching.
    """
    explainer: SHAPExplainerService = getattr(request.app.state, "explainer_service", None)
    if not explainer:
        raise HTTPException(status_code=503, detail="SHAP Explainer service not ready.")

    try:
        result = explainer.explain(
            text=payload.text,
            max_evals=payload.max_evals,
            max_words=payload.max_words
        )
        return ExplainResponse(
            tokens=result["tokens"],
            base_value=result["base_value"],
            prediction_deceptive_prob=result.get("prediction_deceptive_prob"),
            latency_ms=result.get("latency_ms"),
            cached=result.get("cached", False)
        )
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
