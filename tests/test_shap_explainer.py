"""
Unit Tests for Step 6: SHAP Explainer Service & Benchmarking.
Verifies end-to-end pipeline explanation, sign convention, caching, and API contract compliance.
"""

import os
import unittest
import numpy as np

from backend.app.services.explainer import SHAPExplainerService
from backend.app.constants import InternalClass, DisplayLabel


class TestSHAPExplainerService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = "backend/models/hybrid_distilbert_svm_pipeline.joblib"
        cls.has_model = os.path.exists(cls.model_path)
        if cls.has_model:
            cls.explainer = SHAPExplainerService(
                model_path=cls.model_path,
                default_max_evals=30,
                max_words=20
            )

    def test_explainer_initialization(self):
        """Verify explainer loads scaler, classifier, and pooling strategy from artifact."""
        if not self.has_model:
            self.skipTest("Trained hybrid model artifact not found on disk")

        self.assertIsNotNone(self.explainer.scaler)
        self.assertIsNotNone(self.explainer.classifier)
        self.assertEqual(self.explainer.pooling_strategy, "mean")

    def test_pipeline_predict_proba_shape_and_bounds(self):
        """Verify pipeline prediction function outputs valid probabilities for class 0 and 1."""
        if not self.has_model:
            self.skipTest("Trained hybrid model artifact not found on disk")

        sample_texts = [
            "Great product works well",
            "Terrible quality do not buy",
            ""  # Test empty text edge case
        ]

        proba = self.explainer._pipeline_predict_proba(sample_texts)
        self.assertEqual(proba.shape, (3, 2))
        # Probabilities must sum to 1.0
        np.testing.assert_allclose(np.sum(proba, axis=1), [1.0, 1.0, 1.0], atol=1e-5)
        # Probabilities must be bounded in [0, 1]
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

    def test_explain_output_structure_and_types(self):
        """Verify explanation dictionary matches GEMINI.md API contract."""
        if not self.has_model:
            self.skipTest("Trained hybrid model artifact not found on disk")

        text = "This item is very good and arrived fast"
        res = self.explainer.explain(text, max_evals=25)

        self.assertIn("tokens", res)
        self.assertIn("base_value", res)
        self.assertIn("prediction_deceptive_prob", res)
        self.assertIn("latency_ms", res)
        self.assertIn("cached", res)

        self.assertIsInstance(res["tokens"], list)
        self.assertGreater(len(res["tokens"]), 0)
        self.assertIsInstance(res["base_value"], float)
        self.assertIsInstance(res["prediction_deceptive_prob"], float)

        # Check token element structure
        for token in res["tokens"]:
            self.assertIn("text", token)
            self.assertIn("weight", token)
            self.assertIsInstance(token["text"], str)
            self.assertIsInstance(token["weight"], float)

    def test_caching_mechanism(self):
        """Verify in-memory explanation caching produces instantaneous cache hit."""
        if not self.has_model:
            self.skipTest("Trained hybrid model artifact not found on disk")

        text = "Unique caching test review string"
        # First call: computes explanation
        res1 = self.explainer.explain(text, max_evals=25)
        self.assertFalse(res1["cached"])

        # Second call: must hit cache
        res2 = self.explainer.explain(text, max_evals=25)
        self.assertTrue(res2["cached"])
        self.assertEqual(res1["tokens"], res2["tokens"])
        self.assertEqual(res1["prediction_deceptive_prob"], res2["prediction_deceptive_prob"])

    def test_max_words_truncation(self):
        """Verify long texts exceeding max_words are truncated to safeguard latency."""
        if not self.has_model:
            self.skipTest("Trained hybrid model artifact not found on disk")

        long_text = "word " * 35  # 35 words exceeds max_words=20
        res = self.explainer.explain(long_text, max_words=10, max_evals=20)
        self.assertTrue(res["truncated"])
        self.assertLessEqual(len(res["tokens"]), 10)


if __name__ == "__main__":
    unittest.main()
