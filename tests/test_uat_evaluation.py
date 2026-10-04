"""
Unit Tests for Step 9: ISO/IEC 25010:2023 UAT Evaluation Framework.

Verifies:
1. Likert verbal scale conversions.
2. ISO/IEC 25010:2023 quality dimensions (Functional Suitability, Performance Efficiency, Interaction Capability, Explainability).
3. Cohort statistics (30 consumers + 1 domain expert per Fake-SHA thesis scale).
4. Strict separation of concerns: UAT report does not conflate system/UX metrics with model performance metrics.
"""

import os
import json
import unittest
import importlib

# Dynamically import numbered module per repository convention
uat_module = importlib.import_module("notebooks.09_uat_evaluation_framework")
get_verbal_interpretation = uat_module.get_verbal_interpretation
generate_benchmark_respondent_data = uat_module.generate_benchmark_respondent_data
analyze_uat_results = uat_module.analyze_uat_results
EVALUATION_CRITERIA = uat_module.EVALUATION_CRITERIA


class TestUATEvaluation(unittest.TestCase):

    def test_verbal_interpretation_scale(self):
        """Verify Likert scoring scale thresholds."""
        self.assertIn("Strongly Agree", get_verbal_interpretation(4.85))
        self.assertIn("Strongly Agree", get_verbal_interpretation(4.20))
        self.assertIn("Agree", get_verbal_interpretation(4.19))
        self.assertIn("Agree", get_verbal_interpretation(3.40))
        self.assertIn("Neutral", get_verbal_interpretation(3.39))
        self.assertIn("Neutral", get_verbal_interpretation(2.60))
        self.assertIn("Disagree", get_verbal_interpretation(2.59))
        self.assertIn("Disagree", get_verbal_interpretation(1.80))
        self.assertIn("Strongly Disagree", get_verbal_interpretation(1.50))

    def test_iso_25010_dimensions_presence(self):
        """Verify all required ISO/IEC 25010:2023 quality dimensions are defined."""
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

    def test_respondent_cohort_sizes(self):
        """Verify 30 consumers and 1 expert are evaluated per thesis evaluation plan."""
        consumers, expert = generate_benchmark_respondent_data()
        self.assertEqual(len(consumers), 30, "Must have exactly 30 consumer respondents")
        self.assertEqual(expert["role"], "Domain Expert (AI / NLP & Software Engineering)")
        self.assertIn("qualitative_feedback", expert)

    def test_statistical_aggregation_and_bounds(self):
        """Verify computed dimension means and grand means are bounded in [1.0, 5.0]."""
        consumers, expert = generate_benchmark_respondent_data()
        report = analyze_uat_results(consumers, expert)

        summary = report["overall_summary"]
        self.assertGreaterEqual(summary["consumer_grand_mean"], 1.0)
        self.assertLessEqual(summary["consumer_grand_mean"], 5.0)
        self.assertGreaterEqual(summary["expert_grand_mean"], 1.0)
        self.assertLessEqual(summary["expert_grand_mean"], 5.0)

        # Dimension composite check
        for dim_key, dim_res in report["dimensions"].items():
            self.assertIn("dimension_consumer_mean", dim_res)
            self.assertIn("dimension_expert_mean", dim_res)
            self.assertGreaterEqual(dim_res["dimension_consumer_mean"], 1.0)
            self.assertLessEqual(dim_res["dimension_consumer_mean"], 5.0)

    def test_separation_of_concerns_rule(self):
        """
        Verify GEMINI.md rule: Keep model metrics vs. system/UX metrics separate.
        UAT report must not calculate or claim model accuracy, precision, recall, or F1.
        """
        consumers, expert = generate_benchmark_respondent_data()
        report = analyze_uat_results(consumers, expert)

        # Model performance metrics must NOT be in UAT report
        self.assertNotIn("accuracy", report["overall_summary"])
        self.assertNotIn("f1_score", report["overall_summary"])
        self.assertNotIn("precision", report["overall_summary"])
        self.assertNotIn("recall", report["overall_summary"])

        # System quality metrics must be present
        self.assertIn("system_performance_efficiency_metrics", report)
        perf = report["system_performance_efficiency_metrics"]
        self.assertIn("classify_response_time_ms", perf)
        self.assertIn("explain_response_time_ms", perf)

    def test_saved_uat_json_report(self):
        """Verify saved UAT JSON report structure if file exists on disk."""
        report_path = "data/processed/iso_25010_uat_evaluation.json"
        if not os.path.exists(report_path):
            self.skipTest("iso_25010_uat_evaluation.json not yet generated")

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["evaluation_standard"], "ISO/IEC 25010:2023")
        self.assertEqual(data["sample_size"]["total_respondents"], 31)
        self.assertIn("overall_summary", data)


if __name__ == "__main__":
    unittest.main()
