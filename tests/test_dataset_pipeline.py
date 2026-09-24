import os
import shutil
import tempfile
import unittest
from notebooks.preprocess import stratified_split, run_pipeline
from backend.app.constants import InternalClass


class TestDatasetPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stratified_split_balance(self):
        records = [
            {"text": f"Genuine review {i}", "label": InternalClass.GENUINE.value}
            for i in range(20)
        ] + [
            {"text": f"Deceptive review {i}", "label": InternalClass.DECEPTIVE.value}
            for i in range(20)
        ]

        train, val, test = stratified_split(records, val_ratio=0.15, test_ratio=0.15, random_state=42)

        # 40 items total -> 6 test (3 genuine, 3 deceptive), 6 val (3 genuine, 3 deceptive), 28 train (14 genuine, 14 deceptive)
        self.assertEqual(len(test), 6)
        self.assertEqual(len(val), 6)
        self.assertEqual(len(train), 28)

        # Verify class balance in test set
        test_genuine = sum(1 for r in test if r["label"] == InternalClass.GENUINE.value)
        test_deceptive = sum(1 for r in test if r["label"] == InternalClass.DECEPTIVE.value)
        self.assertEqual(test_genuine, 3)
        self.assertEqual(test_deceptive, 3)

    def test_pipeline_execution_and_artifacts(self):
        output_dir = os.path.join(self.temp_dir, "processed")
        metadata = run_pipeline(raw_csv_path=None, output_dir=output_dir)

        self.assertIn("splits", metadata)
        self.assertTrue(os.path.exists(os.path.join(output_dir, "train.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "val.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "metadata.json")))


if __name__ == "__main__":
    unittest.main()
