"""
Tests for Step 3: TF-IDF + LinearSVC Baseline Pipeline.
Verifies pipeline artifact integrity, probability calibration, and metric calculations.
"""

import os
import unittest
import importlib
import joblib
import numpy as np

# Dynamically import numbered module per GEMINI.md naming convention
baseline_module = importlib.import_module("notebooks.03_tfidf_baseline")
evaluate_predictions = baseline_module.evaluate_predictions


class TestTfidfBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact_path = "backend/models/baseline_tfidf_pipeline.joblib"
        cls.has_artifact = os.path.exists(cls.artifact_path)
        if cls.has_artifact:
            cls.pipeline = joblib.load(cls.artifact_path)

    def test_pipeline_artifact_structure(self):
        """Verify that the saved pipeline contains all required components per GEMINI.md."""
        if not self.has_artifact:
            self.skipTest("Baseline artifact not yet trained on disk")

        self.assertIn("vectorizer", self.pipeline)
        self.assertIn("classifier", self.pipeline)
        self.assertIn("best_C", self.pipeline)
        self.assertIn("val_metrics", self.pipeline)

        # Verify n-gram range is 1-2 words per GEMINI.md
        vectorizer = self.pipeline["vectorizer"]
        self.assertEqual(vectorizer.ngram_range, (1, 2))

    def test_probability_calibration_and_prediction(self):
        """Verify that calibrated classifier outputs valid probabilities between 0 and 1."""
        if not self.has_artifact:
            self.skipTest("Baseline artifact not yet trained on disk")

        vectorizer = self.pipeline["vectorizer"]
        classifier = self.pipeline["classifier"]

        sample_texts = [
            "This mouse has great ergonomics, battery lasts for weeks, totally authentic item.",
            "Best product ever best best five stars buy it now super amazing transformed my life!",
        ]
        X = vectorizer.transform(sample_texts)
        proba = classifier.predict_proba(X)
        preds = classifier.predict(X)

        self.assertEqual(proba.shape, (2, 2))
        # Probabilities must sum to 1.0 for each review
        np.testing.assert_allclose(np.sum(proba, axis=1), [1.0, 1.0], atol=1e-5)

        # Predictions must be binary 0 (Genuine) or 1 (Deceptive)
        for p in preds:
            self.assertIn(p, (0, 1))

    def test_evaluate_predictions_math(self):
        """Verify metric calculation helper on known synthetic targets."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])  # 1 TN, 1 FP, 1 FN, 1 TP -> 50% accuracy
        y_proba = np.array([
            [0.9, 0.1],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.2, 0.8]
        ])

        metrics = evaluate_predictions(y_true, y_pred, y_proba)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(metrics["average_confidence"], 0.75, places=2)


if __name__ == "__main__":
    unittest.main()
