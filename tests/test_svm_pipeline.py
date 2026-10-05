"""
Tests for Step 4: DistilBERT-SVM Hybrid Pipeline.
Verifies scaler, calibrated classifier, probability outputs, and artifact structure.
"""

import os
import unittest
import importlib
import joblib
import numpy as np

# Dynamically import numbered module per repository convention
svm_module = importlib.import_module("notebooks.04_train_svm")
evaluate_predictions = svm_module.evaluate_predictions


class TestHybridSVMPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact_path = "backend/models/hybrid_distilbert_svm_pipeline.joblib"
        cls.has_artifact = os.path.exists(cls.artifact_path)
        if cls.has_artifact:
            cls.pipeline = joblib.load(cls.artifact_path)

    def test_pipeline_artifact_structure(self):
        """Verify that the saved hybrid pipeline contains all required components per GEMINI.md."""
        if not self.has_artifact:
            self.skipTest("Hybrid pipeline artifact not yet trained on disk")

        self.assertIn("scaler", self.pipeline)
        self.assertIn("classifier", self.pipeline)
        self.assertIn("raw_svc", self.pipeline)
        self.assertIn("best_C", self.pipeline)
        self.assertIn("val_metrics", self.pipeline)
        self.assertIn("pooling_strategy", self.pipeline)
        self.assertEqual(self.pipeline["feature_dim"], 768)

    def test_scaler_dimension_integrity(self):
        """Verify that the StandardScaler is fit for 768-dimensional DistilBERT embeddings."""
        if not self.has_artifact:
            self.skipTest("Hybrid pipeline artifact not yet trained on disk")

        scaler = self.pipeline["scaler"]
        self.assertEqual(scaler.mean_.shape, (768,))
        self.assertEqual(scaler.scale_.shape, (768,))

    def test_decision_function_and_predictions(self):
        """Verify decision function outputs continuous scores and predictions are binary matching sign per GEMINI.md."""
        if not self.has_artifact:
            self.skipTest("Hybrid pipeline artifact not yet trained on disk")

        scaler = self.pipeline["scaler"]
        classifier = self.pipeline["classifier"]

        # Generate synthetic 768-dim embeddings for 4 test samples
        rng = np.random.RandomState(42)
        dummy_embeddings = rng.randn(4, 768).astype(np.float32)

        X_scaled = scaler.transform(dummy_embeddings)
        scores = classifier.decision_function(X_scaled)
        preds = classifier.predict(X_scaled)

        self.assertEqual(scores.shape, (4,))
        for s, p in zip(scores, preds):
            self.assertIn(p, (0, 1))
            expected = 1 if s > 0 else 0
            self.assertEqual(p, expected)

    def test_evaluate_predictions_function(self):
        """Verify evaluation metric calculations."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])  # TN=1, FP=1, FN=0, TP=2 -> Accuracy=75%
        metrics = evaluate_predictions(y_true, y_pred)

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["recall_deceptive"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])


if __name__ == "__main__":
    unittest.main()
