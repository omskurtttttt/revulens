"""
Unit Tests for Step 9: ISO/IEC 25010:2023 UAT Evaluation Framework.

Verifies:
1. Likert verbal scale conversions matching GEMINI.md Chapter 3 ranges.
2. ISO/IEC 25010:2023 quality dimensions (Functional Suitability, Performance Efficiency, Interaction Capability, Explainability).
3. CSV template generation and response parsing without simulated data.
4. Real latency ingestion from benchmark files.
5. Strict separation of concerns: UAT report does not conflate system/UX metrics with model performance metrics.
"""

import os
import csv
import json
import tempfile
import unittest
import importlib

# Dynamically import numbered module per repository convention
uat_module = importlib.import_module("notebooks.09_uat_evaluation_framework")
get_verbal_interpretation = uat_module.get_verbal_interpretation
create_uat_survey_template_csv = uat_module.create_uat_survey_template_csv
load_uat_responses_from_csv = uat_module.load_uat_responses_from_csv
load_real_latency_benchmarks = uat_module.load_real_latency_benchmarks
analyze_uat_results = uat_module.analyze_uat_results
EVALUATION_CRITERIA = uat_module.EVALUATION_CRITERIA
ALL_ITEM_IDS = uat_module.ALL_ITEM_IDS


class TestUATEvaluation(unittest.TestCase):

    def test_verbal_interpretation_scale(self):
        """Verify Likert scoring scale thresholds per GEMINI.md / Chapter 3."""
        # 4.21 - 5.00: Strongly Agree
        self.assertIn("Strongly Agree", get_verbal_interpretation(5.00))
        self.assertIn("Strongly Agree", get_verbal_interpretation(4.21))
        # 3.41 - 4.20: Agree
        self.assertIn("Agree", get_verbal_interpretation(4.20))
        self.assertIn("Agree", get_verbal_interpretation(3.41))
        # 2.61 - 3.40: Neutral
        self.assertIn("Neutral", get_verbal_interpretation(3.40))
        self.assertIn("Neutral", get_verbal_interpretation(2.61))
        # 1.81 - 2.60: Disagree
        self.assertIn("Disagree", get_verbal_interpretation(2.60))
        self.assertIn("Disagree", get_verbal_interpretation(1.81))
        # 1.00 - 1.80: Strongly Disagree
        self.assertIn("Strongly Disagree", get_verbal_interpretation(1.80))
        self.assertIn("Strongly Disagree", get_verbal_interpretation(1.00))

    def test_iso_25010_dimensions_presence(self):
        """Verify required ISO/IEC 25010:2023 quality dimensions and supplementary explainability."""
        required_dimensions = [
            "functional_suitability",
            "performance_efficiency",
            "interaction_capability",
            "explainability_and_trust"
        ]
        for dim in required_dimensions:
            self.assertIn(dim, EVALUATION_CRITERIA)
            self.assertIn("subcharacteristics", EVALUATION_CRITERIA[dim])
            self.assertGreater(len(EVALUATION_CRITERIA[dim]["subcharacteristics"]), 0)

        # Dimension 4 must be labeled supplementary per GEMINI.md
        self.assertTrue(EVALUATION_CRITERIA["explainability_and_trust"]["is_supplementary"])
        self.assertFalse(EVALUATION_CRITERIA["functional_suitability"]["is_supplementary"])

    def test_csv_template_creation_and_loading(self):
        """Verify CSV template creation and response loading from real CSV structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = os.path.join(tmpdir, "test_template.csv")
            created_path = create_uat_survey_template_csv(template_path)
            self.assertTrue(os.path.exists(created_path))

            # Populate with 2 test rows (1 consumer, 1 expert)
            with open(created_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers = ["respondent_id", "role", "experience_level"] + ALL_ITEM_IDS + ["qualitative_feedback"]
                writer.writerow(headers)
                writer.writerow(["RESP_01", "Consumer", "Weekly"] + ["5" for _ in ALL_ITEM_IDS] + ["Great tool"])
                writer.writerow(["EXPERT_01", "Domain Expert", "Faculty"] + ["4" for _ in ALL_ITEM_IDS] + ["Solid approach"])

            consumers, expert = load_uat_responses_from_csv(created_path)
            self.assertEqual(len(consumers), 1)
            self.assertEqual(consumers[0]["respondent_id"], "RESP_01")
            self.assertIsNotNone(expert)
            self.assertEqual(expert["respondent_id"], "EXPERT_01")

    def test_statistical_aggregation_and_bounds(self):
        """Verify computed dimension means and grand means from response data."""
        # Create test consumer records
        consumers = [
            {
                "respondent_id": f"TEST_RESP_{i}",
                "role": "Consumer",
                "scores": {
                    dim_key: {sub["id"]: 4 for sub in dim_info["subcharacteristics"]}
                    for dim_key, dim_info in EVALUATION_CRITERIA.items()
                }
            }
            for i in range(5)
        ]
        expert = {
            "respondent_id": "TEST_EXPERT_01",
            "role": "Domain Expert",
            "scores": {
                dim_key: {sub["id"]: 5 for sub in dim_info["subcharacteristics"]}
                for dim_key, dim_info in EVALUATION_CRITERIA.items()
            },
            "qualitative_feedback": "Accurate token attribution."
        }

        report = analyze_uat_results(consumers, expert)

        summary = report["overall_summary"]
        self.assertAlmostEqual(summary["consumer_grand_mean"], 4.00, places=2)
        self.assertAlmostEqual(summary["expert_grand_mean"], 5.00, places=2)
        self.assertIn("Agree", summary["consumer_grand_verbal"])
        self.assertIn("Strongly Agree", summary["expert_grand_verbal"])

    def test_separation_of_concerns_rule(self):
        """
        Verify GEMINI.md rule: Keep model metrics vs. system/UX metrics separate.
        UAT report must not calculate or claim model accuracy, precision, recall, or F1.
        """
        consumers = [
            {
                "respondent_id": "TEST_RESP_01",
                "role": "Consumer",
                "scores": {
                    dim_key: {sub["id"]: 5 for sub in dim_info["subcharacteristics"]}
                    for dim_key, dim_info in EVALUATION_CRITERIA.items()
                }
            }
        ]
        report = analyze_uat_results(consumers, None)

        # Model performance metrics must NOT be in UAT report
        self.assertNotIn("accuracy", report["overall_summary"])
        self.assertNotIn("f1_score", report["overall_summary"])
        self.assertNotIn("precision", report["overall_summary"])
        self.assertNotIn("recall", report["overall_summary"])

        # System quality metrics must be present
        self.assertIn("system_performance_efficiency_metrics", report)

    def test_load_real_latency_benchmarks(self):
        """Verify latency is read from real benchmark files when available."""
        perf = load_real_latency_benchmarks()
        self.assertIn("explain_response_time_ms", perf)
        self.assertIn("source", perf)


if __name__ == "__main__":
    unittest.main()
