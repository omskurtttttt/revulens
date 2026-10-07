"""
Concurrency & Thread-Safety Tests for RevuLens FastAPI Service.

Adheres strictly to GEMINI.md TODO 3:
1. Verifies /classify is NOT blocked while a heavy /explain request is running.
2. Verifies /health and / respond immediately on the event loop while /explain is running in the worker threadpool.
3. Verifies SHAPExplainerService and in-memory cache are thread-safe under concurrent access.
"""

import os
import time
import unittest
import concurrent.futures
from typing import Dict, Any

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.constants import InternalClass, DisplayLabel


class TestBackendConcurrency(unittest.TestCase):
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

    def test_classify_while_explain_runs(self):
        """
        Verify GEMINI.md requirement:
        Classification is never held up by explanation.
        A /classify request sent while /explain is running must finish promptly
        and complete before /explain finishes.
        """
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        explain_text = "This product arrived in pristine condition, the packaging was excellent and shipping was very prompt."
        classify_text = "Defective item broke immediately upon first use."

        explain_result = {}
        classify_result = {}
        events = []

        def run_explain():
            events.append(("explain_start", time.perf_counter()))
            res = self.client.post("/explain", json={"text": explain_text, "max_evals": 35})
            events.append(("explain_end", time.perf_counter()))
            explain_result["status"] = res.status_code
            explain_result["data"] = res.json()

        def run_classify():
            # Tiny sleep to ensure run_explain has started executing first
            time.sleep(0.08)
            events.append(("classify_start", time.perf_counter()))
            res = self.client.post("/classify", json={"text": classify_text})
            events.append(("classify_end", time.perf_counter()))
            classify_result["status"] = res.status_code
            classify_result["data"] = res.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_explain = executor.submit(run_explain)
            fut_classify = executor.submit(run_classify)
            fut_explain.result()
            fut_classify.result()

        # Both requests must succeed
        self.assertEqual(explain_result["status"], 200)
        self.assertEqual(classify_result["status"], 200)

        # Classify must return valid response
        self.assertIn(classify_result["data"]["label"], (InternalClass.GENUINE.value, InternalClass.DECEPTIVE.value))
        self.assertIn("display_label", classify_result["data"])

        # Explain must return valid tokens
        self.assertGreater(len(explain_result["data"]["tokens"]), 0)

        # Extract timestamps
        event_dict = dict(events)
        classify_duration = event_dict["classify_end"] - event_dict["classify_start"]

        # Classify must finish before explain finishes
        self.assertLess(
            event_dict["classify_end"],
            event_dict["explain_end"],
            f"Classify (finished at {event_dict['classify_end']:.3f}) should finish before Explain (finished at {event_dict['explain_end']:.3f})"
        )

        # Classify latency should be fast (sub-second on threadpool, not blocked by full explanation time)
        self.assertLess(
            classify_duration,
            1.5,
            f"Classify took {classify_duration:.3f}s; expected prompt threadpool dispatch"
        )

    def test_health_check_non_blocking_during_explain(self):
        """
        Verify /health responds immediately on the event loop while a heavy /explain runs.
        """
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        explain_text = "The audio quality is crystal clear with deep bass and great wireless range."
        results = {}

        def run_heavy_explain():
            res = self.client.post("/explain", json={"text": explain_text, "max_evals": 35})
            results["explain"] = res.status_code

        def run_health():
            time.sleep(0.05)
            t0 = time.perf_counter()
            res = self.client.get("/health")
            elapsed = time.perf_counter() - t0
            results["health_status"] = res.status_code
            results["health_time"] = elapsed

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(run_heavy_explain)
            fut2 = executor.submit(run_health)
            fut1.result()
            fut2.result()

        self.assertEqual(results["explain"], 200)
        self.assertEqual(results["health_status"], 200)
        self.assertLess(results["health_time"], 0.5, "Health check should not be blocked by explain")

    def test_concurrent_identical_explain_cache_safety(self):
        """
        Verify concurrent explain requests for the same text do not cause race conditions
        in the SHAP cache or explainer instance.
        """
        if not self.has_model:
            self.skipTest("Hybrid model pipeline not found on disk")

        text = "Super fast delivery authentic item five stars."
        responses = []

        def call_explain():
            res = self.client.post("/explain", json={"text": text, "max_evals": 25})
            return res.status_code, res.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(call_explain) for _ in range(3)]
            for fut in futures:
                status, data = fut.result()
                self.assertEqual(status, 200)
                responses.append(data)

        # All 3 must succeed with identical token count
        token_lens = [len(r["tokens"]) for r in responses]
        self.assertEqual(len(set(token_lens)), 1)


if __name__ == "__main__":
    unittest.main()
