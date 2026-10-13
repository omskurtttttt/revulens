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
        import csv

        # Create minimal synthetic raw CSVs in self.temp_dir so test is fully self-contained
        # and passes in fresh CI environments where raw dataset dumps are gitignored.
        salminen_csv = os.path.join(self.temp_dir, "sample_salminen.csv")
        with open(salminen_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "rating", "label", "text_"])
            writer.writeheader()
            for i in range(20):
                writer.writerow({"category": "Electronics", "rating": "5.0", "label": "OR", "text_": f"Sample genuine product review {i}"})
                writer.writerow({"category": "Electronics", "rating": "1.0", "label": "CG", "text_": f"Sample deceptive product review {i}"})

        firecs_train_csv = os.path.join(self.temp_dir, "sample_firecs_train.csv")
        with open(firecs_train_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["review", "label"])
            writer.writeheader()
            writer.writerow({"review": "Maganda ang item mabilis dumating", "label": "positive"})

        firecs_test_csv = os.path.join(self.temp_dir, "sample_firecs_test.csv")
        with open(firecs_test_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["review", "label"])
            writer.writeheader()
            writer.writerow({"review": "Sira ang item pagdating sa akin", "label": "negative"})

        output_dir = os.path.join(self.temp_dir, "processed")
        metadata = run_pipeline(
            raw_salminen_path=salminen_csv,
            firecs_train_path=firecs_train_csv,
            firecs_test_path=firecs_test_csv,
            output_dir=output_dir,
            random_state=42
        )

        self.assertIn("splits", metadata)
        self.assertEqual(metadata["dataset_rules_version"], "GEMINI.md (80/10/10 split, fixed seed 42)")
        self.assertTrue(os.path.exists(os.path.join(output_dir, "train.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "val.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "firecs_exploratory.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "metadata.json")))


if __name__ == "__main__":
    unittest.main()
