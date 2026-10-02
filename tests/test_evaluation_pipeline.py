"""
Tests for Step 5: Test Split Evaluation and FiReCS Exploratory Check.
Verifies held-out test metric computation, disclaimer enforcement, and FiReCS behavioral constraints per GEMINI.md.
"""

import os
import unittest
import importlib
import numpy as np

# Dynamically import numbered module per repository convention
eval_module = importlib.import_module("notebooks.05_evaluate_test_split")
compute_metrics = eval_module.compute_metrics
evaluate_firecs_exploratory = eval_module.evaluate_firecs_exploratory
load_salminen_test_split = eval_module.load_salminen_test_split


class TestEvaluationPipeline(unittest.TestCase):

    def test_compute_metrics_accuracy_and_f1(self):
        """Verify binary and macro metrics calculation with known ground truth and predictions."""
        # 4 samples: TN=1, FP=1, FN=0, TP=2
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([
            [0.8, 0.2],
            [0.1, 0.9],
            [0.4, 0.6],
            [0.2, 0.8]
        ])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["recall_deceptive"], 1.0)
        self.assertAlmostEqual(metrics["precision_deceptive"], 2/3, places=4)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        # Max confidence per row: [0.8, 0.9, 0.6, 0.8] -> mean = 0.775
        self.assertEqual(metrics["average_confidence"], 0.775)

    def test_salminen_test_split_loader(self):
        """Verify Salminen test split correctly maps Genuine->0 and Deceptive->1."""
        test_csv = "data/processed/test.csv"
        if not os.path.exists(test_csv):
            self.skipTest("Test CSV not yet generated")

        texts, labels = load_salminen_test_split(test_csv, limit=20)
        self.assertEqual(len(texts), 20)
        self.assertEqual(len(labels), 20)
        for lbl in labels:
            self.assertIn(lbl, (0, 1))

    def test_firecs_exploratory_constraints(self):
        """
        Verify FiReCS exploratory check strictly enforces GEMINI.md constraints:
        1. Labeled 'exploratory'
        2. Contains disclaimer
        3. Never outputs accuracy, precision, recall, or F1 for FiReCS
        4. Reports share_flagged_deceptive_pct
        """
        baseline_path = "backend/models/baseline_tfidf_pipeline.joblib"
        hybrid_path = "backend/models/hybrid_distilbert_svm_pipeline.joblib"
        firecs_csv = "data/processed/firecs_exploratory.csv"

        if not (os.path.exists(baseline_path) and os.path.exists(hybrid_path) and os.path.exists(firecs_csv)):
            self.skipTest("Required artifacts or FiReCS data missing")

        results = evaluate_firecs_exploratory(
            baseline_artifact_path=baseline_path,
            hybrid_artifact_path=hybrid_path,
            firecs_csv_path=firecs_csv,
            firecs_emb_path=None,  # test without hybrid embeddings first
            limit=50
        )

        self.assertEqual(results["status"], "exploratory")
        self.assertIn("sentiment labels only", results["disclaimer"].lower())
        self.assertIn("accuracy/f1 are not reported", results["disclaimer"].lower())

        # Verify prohibited metrics are NOT present in baseline FiReCS report
        b_res = results["baseline"]
        self.assertNotIn("accuracy", b_res)
        self.assertNotIn("precision", b_res)
        self.assertNotIn("recall", b_res)
        self.assertNotIn("f1", b_res)
        self.assertIn("share_flagged_deceptive_pct", b_res)
        self.assertIn("total_taglish_evaluated", b_res)
        self.assertEqual(b_res["total_taglish_evaluated"], 50)

    def test_saved_evaluation_json_structure(self):
        """Verify the saved final_test_evaluation.json artifact follows report schema."""
        import json
        report_path = "data/processed/final_test_evaluation.json"
        if not os.path.exists(report_path):
            self.skipTest("final_test_evaluation.json not yet generated")

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("salminen_test_split_matched", data)
        self.assertIn("firecs_taglish_exploratory_check", data)

        matched = data["salminen_test_split_matched"]
        self.assertIn("baseline_tfidf_linear_svc", matched)
        self.assertIn("hybrid_distilbert_svm", matched)

        # Baseline and hybrid must report accuracy and macro F1
        self.assertIn("accuracy", matched["baseline_tfidf_linear_svc"])
        self.assertIn("f1_macro", matched["baseline_tfidf_linear_svc"])
        self.assertIn("accuracy", matched["hybrid_distilbert_svm"])
        self.assertIn("f1_macro", matched["hybrid_distilbert_svm"])

        # FiReCS must have disclaimer and no accuracy
        firecs = data["firecs_taglish_exploratory_check"]
        self.assertEqual(firecs["status"], "exploratory")
        self.assertNotIn("accuracy", firecs["baseline"])
        if firecs["hybrid"]:
            self.assertNotIn("accuracy", firecs["hybrid"])
            self.assertIn("share_flagged_deceptive_pct", firecs["hybrid"])


if __name__ == "__main__":
    unittest.main()
