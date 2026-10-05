"""
SHAP Explanation Engine for RevuLens (Step 6).

Adheres strictly to GEMINI.md:
1. Explains the end-to-end pipeline:
   text -> normalize -> DistilBERT (frozen) -> StandardScaler -> Calibrated LinearSVC -> P(Deceptive)
2. Uses SHAP's text masker approach.
3. Sign convention: explains P(Deceptive).
   - Positive weight (> 0) pushes toward Deceptive.
   - Negative weight (< 0) pushes toward Genuine.
4. Token-level output format:
   { "tokens": [{"text": str, "weight": float}], "base_value": float, "prediction": float }
5. Non-blocking & cached: Includes in-memory caching and configurable max_evals / max_words limits.
"""

import os
import re
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple

import joblib
import numpy as np

from backend.app.preprocessing import normalize_text
from backend.app.services.embedding import DistilBERTEmbeddingExtractor


class SHAPExplainerService:
    """
    Post-hoc explainability service computing token-level SHAP attributions
    for the DistilBERT-SVM hybrid pipeline.
    """

    def __init__(
        self,
        model_path: str = "backend/models/hybrid_distilbert_svm_pipeline.joblib",
        extractor: Optional[DistilBERTEmbeddingExtractor] = None,
        default_max_evals: int = 100,
        max_words: int = 100,
        cache_size: int = 128
    ):
        """
        Initialize the explainer service.
        Loads the trained hybrid pipeline and configures SHAP text masker.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hybrid pipeline model artifact not found at: {model_path}")

        self.model_path = model_path
        self.default_max_evals = default_max_evals
        self.max_words = max_words
        self.cache_size = cache_size

        # In-memory explanation cache: {cache_key: explanation_dict}
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Load trained hybrid pipeline components
        pipeline = joblib.load(model_path)
        self.scaler = pipeline["scaler"]
        self.classifier = pipeline["classifier"]
        self.pooling_strategy = pipeline.get("pooling_strategy", "mean")

        # Reuse or initialize DistilBERT embedding extractor
        if extractor is not None:
            self.extractor = extractor
        else:
            self.extractor = DistilBERTEmbeddingExtractor(
                pooling_strategy=self.pooling_strategy,
                max_length=128
            )

        # Lazy-initialized SHAP explainer instance
        self._explainer = None

    def _pipeline_predict_proba(self, texts: np.ndarray | List[str]) -> np.ndarray:
        """
        Black-box prediction function for SHAP:
        Maps masked raw text batches -> DistilBERT embeddings -> Scaler -> Calibrated Probabilities.
        Returns 2D array of shape (N, 2) where column 1 is P(Deceptive).
        """
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()

        # Handle empty/masked text entries safely
        cleaned_batch = []
        for t in texts:
            if not t or not isinstance(t, str) or not t.strip():
                # Provide a neutral single space when all words are masked out
                cleaned_batch.append(" ")
            else:
                norm = normalize_text(t)
                cleaned_batch.append(norm if norm else " ")

        # 1. DistilBERT frozen contextual embedding
        embeddings = self.extractor.extract(
            cleaned_batch,
            batch_size=64,
            pooling_strategy=self.pooling_strategy
        )

        # 2. StandardScaler fit on train
        scaled = self.scaler.transform(embeddings)

        # 3. Probabilities (or logistic sigmoid over decision_function for LinearSVC)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(scaled)

        # Map LinearSVC decision function to pseudo-probabilities [P(Genuine), P(Deceptive)]
        decision = self.classifier.decision_function(scaled)
        p_deceptive = 1.0 / (1.0 + np.exp(-np.clip(decision, -50.0, 50.0)))
        p_genuine = 1.0 - p_deceptive
        return np.column_stack([p_genuine, p_deceptive])

    def _get_explainer(self):
        """Lazy initialization of SHAP Explainer with Text masker."""
        if self._explainer is None:
            import shap

            # Text masker that splits on whitespace while preserving word tokens
            masker = shap.maskers.Text(tokenizer=r"\s+")
            self._explainer = shap.Explainer(
                self._pipeline_predict_proba,
                masker=masker,
                output_names=["Genuine", "Deceptive"]
            )
        return self._explainer

    def _make_cache_key(self, text: str, max_evals: int) -> str:
        """Generate a deterministic MD5 hash key for caching explanations."""
        payload = f"{text.strip().lower()}::max_evals={max_evals}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def explain(
        self,
        text: str,
        max_evals: Optional[int] = None,
        max_words: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute word-level SHAP attributions for the input review text.

        Args:
            text: Raw input review text selected by user.
            max_evals: Maximum model evaluations for SHAP permutation (default: self.default_max_evals).
            max_words: Maximum word count to explain (longer texts are truncated for latency).

        Returns:
            Dictionary following the GEMINI.md API contract:
            {
                "tokens": [{"text": str, "weight": float}],
                "base_value": float,
                "prediction_deceptive_prob": float,
                "latency_ms": float,
                "cached": bool,
                "truncated": bool
            }
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            return {
                "tokens": [],
                "base_value": 0.5,
                "prediction_deceptive_prob": 0.5,
                "latency_ms": 0.0,
                "cached": False,
                "truncated": False
            }

        # 1. Truncate text if exceeding max_words limit (latency control)
        effective_max_words = max_words or self.max_words
        words = text.strip().split()
        truncated = False
        if len(words) > effective_max_words:
            text = " ".join(words[:effective_max_words])
            truncated = True

        # 2. Check cache
        evals_budget = max_evals or self.default_max_evals
        cache_key = self._make_cache_key(text, evals_budget)
        if cache_key in self._cache:
            cached_res = self._cache[cache_key].copy()
            cached_res["cached"] = True
            cached_res["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            return cached_res

        # 3. Compute explanation via SHAP
        explainer = self._get_explainer()
        explanation = explainer([text], max_evals=evals_budget)

        # 4. Extract token texts and weights for P(Deceptive) (class index 1)
        raw_tokens = explanation.data[0]
        # Shape of values: (num_tokens, 2) -> column 1 is Deceptive
        raw_values = explanation.values[0]
        if raw_values.ndim == 2:
            deceptive_weights = raw_values[:, 1]
        else:
            deceptive_weights = raw_values

        # Base value for Deceptive class
        raw_base = explanation.base_values[0]
        if isinstance(raw_base, (np.ndarray, list)):
            base_value = float(raw_base[1])
        else:
            base_value = float(raw_base)

        tokens_list = []
        for t, w in zip(raw_tokens, deceptive_weights):
            clean_t = t.strip()
            if clean_t:
                tokens_list.append({
                    "text": clean_t,
                    "weight": round(float(w), 4)
                })

        # Calculate final model prediction probability
        prob_pred = self._pipeline_predict_proba([text])[0]
        p_deceptive = float(prob_pred[1])

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        result = {
            "tokens": tokens_list,
            "base_value": round(base_value, 4),
            "prediction_deceptive_prob": round(p_deceptive, 4),
            "latency_ms": elapsed_ms,
            "cached": False,
            "truncated": truncated
        }

        # Store in cache (FIFO eviction if capacity reached)
        if len(self._cache) >= self.cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[cache_key] = result.copy()

        return result
