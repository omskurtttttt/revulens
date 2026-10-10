import os
import shutil
import tempfile
import unittest
import importlib

# Dynamically import numbered module per repository convention
preprocess_module = importlib.import_module("notebooks.01_preprocess")
stratified_split_80_10_10 = preprocess_module.stratified_split_80_10_10
run_pipeline = preprocess_module.run_pipeline

from backend.app.constants import InternalClass


class TestDatasetPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stratified_split_80_10_10_balance(self):
        records = [
            {"text": f"Genuine review {i}", "label": InternalClass.GENUINE.value}
            for i in range(50)
        ] + [
            {"text": f"Deceptive review {i}", "label": InternalClass.DECEPTIVE.value}
            for i in range(50)
        ]

        train, val, test = stratified_split_80_10_10(records, val_ratio=0.10, test_ratio=0.10, random_state=42)

        # 100 items total -> 10 test (5 genuine, 5 deceptive), 10 val (5 genuine, 5 deceptive), 80 train (40 genuine, 40 deceptive)
        self.assertEqual(len(test), 10)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(train), 80)

        # Verify class balance in test set
        test_genuine = sum(1 for r in test if r["label"] == InternalClass.GENUINE.value)
        test_deceptive = sum(1 for r in test if r["label"] == InternalClass.DECEPTIVE.value)
        self.assertEqual(test_genuine, 5)
        self.assertEqual(test_deceptive, 5)

    def test_pipeline_execution_and_artifacts(self):
        output_dir = os.path.join(self.temp_dir, "processed")
        metadata = run_pipeline(output_dir=output_dir)

        self.assertIn("splits", metadata)
        self.assertEqual(metadata["dataset_rules_version"], "GEMINI.md (80/10/10 split, fixed seed 42)")
        self.assertTrue(os.path.exists(os.path.join(output_dir, "train.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "val.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "firecs_exploratory.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "metadata.json")))


if __name__ == "__main__":
    unittest.main()
