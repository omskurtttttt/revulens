"""
Integration Tests for RevuLens FastAPI Backend (Step 7).

Verifies:
1. GET /health
2. POST /classify (contract schema: label, display_label, confidence)
3. POST /explain (contract schema: tokens, base_value)
4. ISO/IEC 25010 response timer header
5. Caching and non-blocking behavior
"""

import os
import unittest
from fastapi.testclient import TestClient

from backend.app.main import app
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
        { label, display_label, confidence }
        """
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        payload = {
            "text": "The delivery was surprisingly fast and the shoes are comfortable and fit well."
        }
        response = self.client.post("/classify", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        # Verify required keys
        self.assertIn("label", data)
        self.assertIn("display_label", data)
        self.assertIn("confidence", data)

        # Verify allowed values per constants.py
        self.assertIn(data["label"], (InternalClass.GENUINE.value, InternalClass.DECEPTIVE.value))
        self.assertIn(data["display_label"], (DisplayLabel.LIKELY_GENUINE.value, DisplayLabel.POTENTIALLY_DECEPTIVE.value))

        # Verify exact mapping consistency
        if data["label"] == InternalClass.GENUINE.value:
            self.assertEqual(data["display_label"], DisplayLabel.LIKELY_GENUINE.value)
        else:
            self.assertEqual(data["display_label"], DisplayLabel.POTENTIALLY_DECEPTIVE.value)

        # Verify confidence range [0.0, 1.0]
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)

        # Verify ISO/IEC 25010 response timer header
        self.assertIn("x-response-time-ms", response.headers)

    def test_explain_contract_and_values(self):
        """
        Verify POST /explain satisfies GEMINI.md contract:
        { tokens: [{ text, weight }], base_value }
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
        self.assertIn("base_value", data)
        self.assertIsInstance(data["tokens"], list)
        self.assertGreater(len(data["tokens"]), 0)

        # Verify token format
        for tok in data["tokens"]:
            self.assertIn("text", tok)
            self.assertIn("weight", tok)
            self.assertIsInstance(tok["text"], str)
            self.assertIsInstance(tok["weight"], float)

        # Base value in [0, 1] probability range
        self.assertGreaterEqual(data["base_value"], 0.0)
        self.assertLessEqual(data["base_value"], 1.0)

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

    def test_classify_invalid_empty_input(self):
        """Verify API handles empty text with validation error."""
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")
        response = self.client.post("/classify", json={"text": ""})
        self.assertEqual(response.status_code, 422)  # Pydantic validation error


if __name__ == "__main__":
    unittest.main()
