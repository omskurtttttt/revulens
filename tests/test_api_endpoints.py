"""
Integration Tests for RevuLens FastAPI Backend (GEMINI.md).

Verifies:
1. GET /health
2. POST /classify (contract schema: label, display_label, truncated, latency_ms; ZERO confidence score)
3. POST /explain (contract schema: tokens, truncated, latency_ms, cached; ZERO base value/probability)
4. ISO/IEC 25010 response timer header
5. Caching and non-blocking behavior
6. Input handling:
   - Blank and whitespace-only text rejected with HTTP 422
   - Text exactly at MAX_WORDS (truncated: False)
   - Text one word over MAX_WORDS (truncated: True)
   - Text over MAX_CHARS rejected with HTTP 413 (Payload Too Large)
"""

import os
import unittest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import settings
from backend.app.constants import InternalClass, DisplayLabel


class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = "backend/models/hybrid_distilbert_svm_pipeline.joblib"
        cls.has_model = os.path.exists(model_path)
        if not cls.has_model:
            return

        # Initialize TestClient within lifespan context
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "client"):
            cls.client.__exit__(None, None, None)

    def test_root_endpoint(self):
        """Verify root endpoint returns API information."""
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("service", data)
        self.assertIn("endpoints", data)

    def test_health_check_endpoint(self):
        """Verify /health returns 200 and reports model loaded."""
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertIn("device", data)

    def test_classify_contract_and_values(self):
        """
        Verify POST /classify satisfies GEMINI.md contract:
        { label, display_label, truncated, latency_ms }
        Never exposes confidence score, probability, or raw decision score.
        """
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        payload = {
            "text": "The delivery was surprisingly fast and the shoes are comfortable and fit well."
        }
        response = self.client.post("/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        # Verify required keys per GEMINI.md
        self.assertIn("label", data)
        self.assertIn("display_label", data)
        self.assertIn("truncated", data)
        self.assertIn("latency_ms", data)

        # STRICT: No confidence score, probability, or decision score anywhere
        self.assertNotIn("confidence", data)
        self.assertNotIn("decision_score", data)
        self.assertNotIn("probability", data)

        # Verify allowed values per constants.py
        self.assertIn(data["label"], (InternalClass.GENUINE.value, InternalClass.DECEPTIVE.value))
        self.assertIn(data["display_label"], (DisplayLabel.LIKELY_GENUINE.value, DisplayLabel.POTENTIALLY_DECEPTIVE.value))

        # Verify exact mapping consistency
        if data["label"] == InternalClass.GENUINE.value:
            self.assertEqual(data["display_label"], DisplayLabel.LIKELY_GENUINE.value)
        else:
            self.assertEqual(data["display_label"], DisplayLabel.POTENTIALLY_DECEPTIVE.value)

        # Text was short; truncated must be False
        self.assertFalse(data["truncated"])

        # Verify ISO/IEC 25010 response timer header
        self.assertIn("x-response-time-ms", response.headers)

    def test_explain_contract_and_values(self):
        """
        Verify POST /explain satisfies GEMINI.md contract:
        { tokens: [{ text, weight }], truncated, latency_ms, cached }
        Never exposes base_value, confidence, or probability.
        """
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        payload = {
            "text": "Great packaging and works as described",
            "max_evals": 30
        }
        response = self.client.post("/explain", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("tokens", data)
        self.assertIn("truncated", data)
        self.assertIn("latency_ms", data)
        self.assertIn("cached", data)

        # STRICT: No base value or probability exposed to user
        self.assertNotIn("base_value", data)
        self.assertNotIn("prediction_deceptive_prob", data)
        self.assertNotIn("confidence", data)

        self.assertIsInstance(data["tokens"], list)
        self.assertGreater(len(data["tokens"]), 0)

        # Verify token format
        for tok in data["tokens"]:
            self.assertIn("text", tok)
            self.assertIn("weight", tok)
            self.assertIsInstance(tok["text"], str)
            self.assertIsInstance(tok["weight"], float)

        self.assertFalse(data["truncated"])

    def test_explain_caching_behavior(self):
        """Verify subsequent /explain call with identical text returns cached=True."""
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        payload = {
            "text": "Unique test text string for API caching verification",
            "max_evals": 25
        }
        # First call
        res1 = self.client.post("/explain", json=payload)
        self.assertEqual(res1.status_code, 200)
        self.assertFalse(res1.json().get("cached", False))

        # Second call
        res2 = self.client.post("/explain", json=payload)
        self.assertEqual(res2.status_code, 200)
        self.assertTrue(res2.json().get("cached", False))
        self.assertEqual(res1.json()["tokens"], res2.json()["tokens"])

    def test_classify_invalid_blank_and_whitespace_input(self):
        """Verify API rejects blank and whitespace-only text with HTTP 422 and clear error."""
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        # Empty string
        res_empty = self.client.post("/classify", json={"text": ""})
        self.assertEqual(res_empty.status_code, 422)

        # Whitespace-only string
        res_spaces = self.client.post("/classify", json={"text": "     \n\t  "})
        self.assertEqual(res_spaces.status_code, 422)
        self.assertIn("blank or whitespace", res_spaces.text.lower())

    def test_explain_invalid_blank_and_whitespace_input(self):
        """Verify /explain rejects blank and whitespace-only text with HTTP 422."""
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        res_empty = self.client.post("/explain", json={"text": ""})
        self.assertEqual(res_empty.status_code, 422)

        res_spaces = self.client.post("/explain", json={"text": "   "})
        self.assertEqual(res_spaces.status_code, 422)
        self.assertIn("blank or whitespace", res_spaces.text.lower())

    def test_text_exactly_at_max_words(self):
        """Verify text with exactly MAX_WORDS words is processed with truncated=False."""
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        # Exactly settings.MAX_WORDS (default 100) words
        text_100_words = " ".join([f"word{i}" for i in range(settings.MAX_WORDS)])
        response = self.client.post("/classify", json={"text": text_100_words})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["truncated"], "Expected truncated=False for text at exactly MAX_WORDS")

    def test_text_one_word_over_max_words(self):
        """Verify text with MAX_WORDS + 1 words is truncated with truncated=True."""
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        # Exactly settings.MAX_WORDS + 1 (default 101) words
        text_101_words = " ".join([f"word{i}" for i in range(settings.MAX_WORDS + 1)])
        response = self.client.post("/classify", json={"text": text_101_words})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["truncated"], "Expected truncated=True for text with MAX_WORDS + 1 words")

    def test_text_over_max_chars_rejected_with_413(self):
        """Verify text exceeding MAX_CHARS is rejected with HTTP 413 Payload Too Large."""
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        # Exceeds settings.MAX_CHARS (default 2000)
        oversized_text = "a" * (settings.MAX_CHARS + 50)
        res_classify = self.client.post("/classify", json={"text": oversized_text})
        self.assertEqual(res_classify.status_code, 413)
        self.assertIn("maximum length", res_classify.text.lower())

        res_explain = self.client.post("/explain", json={"text": oversized_text})
        self.assertEqual(res_explain.status_code, 413)
        self.assertIn("maximum length", res_explain.text.lower())


if __name__ == "__main__":
    unittest.main()
