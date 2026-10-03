"""
Classification Service for RevuLens (Step 7).

Adheres strictly to GEMINI.md:
1. Normalizes text using the shared normalize_text function from backend.app.preprocessing.
2. Extracts frozen DistilBERT embedding (mean pooling).
3. Standardizes features with StandardScaler fit on train.
4. Predicts calibrated probability via CalibratedClassifierCV.
5. Returns { label, display_label, confidence } with sub-150ms execution on CPU.
"""

import os
import time
from typing import Dict, Any, Optional

import joblib
import numpy as np

from backend.app.preprocessing import normalize_text
from backend.app.constants import InternalClass, get_display_label
from backend.app.services.embedding import DistilBERTEmbeddingExtractor


class ClassificationService:
    """
    Handles real-time inference for review classification.
    Optimized for high-throughput, low-latency execution.
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
        Classify raw review text into { label, display_label, confidence }.

        Pipeline:
        text -> normalize_text -> DistilBERT frozen embedding -> StandardScaler -> CalibratedClassifierCV -> (label, display_label, confidence)
        """
        start_time = time.perf_counter()

        # 1. Normalize text using shared preprocessing function
        cleaned = normalize_text(text)
        if not cleaned:
            # Fallback for empty or non-text selections
            return {
                "label": InternalClass.GENUINE.value,
                "display_label": get_display_label(InternalClass.GENUINE.value),
                "confidence": 0.5000,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2)
            }

        # 2. Extract frozen contextual embedding
        emb = self.extractor.extract([cleaned], batch_size=1, pooling_strategy=self.pooling_strategy)

        # 3. Standardize features
        scaled = self.scaler.transform(emb)

        # 4. Calibrated probabilities: [P(Genuine), P(Deceptive)]
        proba = self.classifier.predict_proba(scaled)[0]
        p_genuine = float(proba[0])
        p_deceptive = float(proba[1])

        # 5. Derive label and display label
        if p_deceptive >= 0.5:
            internal_label = InternalClass.DECEPTIVE.value
            confidence = p_deceptive
        else:
            internal_label = InternalClass.GENUINE.value
            confidence = p_genuine

        display_lbl = get_display_label(internal_label)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "label": internal_label,
            "display_label": display_lbl,
            "confidence": round(confidence, 4),
            "latency_ms": elapsed_ms
        }
