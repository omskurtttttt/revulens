"""
Classification Service for RevuLens.

Adheres strictly to GEMINI.md:
1. Normalizes text using the shared normalize_text function from backend.app.preprocessing.
2. Extracts frozen DistilBERT embedding (mean pooling).
3. Standardizes features with StandardScaler fit on train only.
4. LinearSVC decision function sign decides class (positive -> Deceptive, negative -> Genuine).
5. No confidence score, probability, or percentage is computed or exposed.
6. Blank or empty text raises an error; never returns a guessed label.
"""

import os
import time
import logging
from typing import Dict, Any, Optional

import joblib
import numpy as np

from backend.app.preprocessing import normalize_text
from backend.app.constants import InternalClass, get_display_label
from backend.app.services.embedding import DistilBERTEmbeddingExtractor

logger = logging.getLogger("revulens")


class ClassificationService:
    """
    Handles real-time inference for review classification.
    Optimized for high-throughput, low-latency execution (<150 ms on CPU).
    """

    def __init__(
        self,
        model_path: str = "backend/models/hybrid_distilbert_svm_pipeline.joblib",
        extractor: Optional[DistilBERTEmbeddingExtractor] = None
    ):
        """Initialize classifier service and load hybrid pipeline components."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hybrid model pipeline artifact not found at: {model_path}")

        self.model_path = model_path
        pipeline = joblib.load(model_path)
        self.scaler = pipeline["scaler"]
        self.classifier = pipeline["classifier"]
        self.pooling_strategy = pipeline.get("pooling_strategy", "mean")

        # Share or initialize frozen DistilBERT extractor
        if extractor is not None:
            self.extractor = extractor
        else:
            self.extractor = DistilBERTEmbeddingExtractor(
                pooling_strategy=self.pooling_strategy,
                max_length=128
            )

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify raw review text into { label, display_label, latency_ms }.

        Pipeline:
        text -> normalize_text -> DistilBERT frozen embedding -> StandardScaler -> LinearSVC -> (label, display_label)
        Sign of decision_function decides class: > 0 -> Deceptive, <= 0 -> Genuine.
        No confidence score is ever computed or returned.
        """
        start_time = time.perf_counter()

        # 1. Normalize text using shared preprocessing function
        cleaned = normalize_text(text)
        if not cleaned:
            raise ValueError("Review text cannot be blank or whitespace-only.")

        # 2. Extract frozen contextual embedding
        emb = self.extractor.extract([cleaned], batch_size=1, pooling_strategy=self.pooling_strategy)

        # 3. Standardize features
        scaled = self.scaler.transform(emb)

        # 4. Sign of SVM decision function strictly decides class per GEMINI.md
        if hasattr(self.classifier, "decision_function"):
            decision_score = float(self.classifier.decision_function(scaled)[0])
        else:
            # Fallback for models without decision_function (e.g., standard linear weights)
            decision_score = float(self.classifier.predict(scaled)[0])

        internal_label = InternalClass.DECEPTIVE.value if decision_score > 0 else InternalClass.GENUINE.value
        display_lbl = get_display_label(internal_label)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "label": internal_label,
            "display_label": display_lbl,
            "decision_score": decision_score,
            "latency_ms": elapsed_ms
        }
