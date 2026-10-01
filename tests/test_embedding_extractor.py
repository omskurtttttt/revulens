"""
Tests for DistilBERTEmbeddingExtractor service.
Verifies frozen weights, shape integrity (768-dim), and both mean and CLS pooling.
"""

import unittest
import numpy as np


class TestDistilBERTEmbeddingExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import inside setUpClass to allow tests to run only when torch/transformers are installed
        try:
            from backend.app.services.embedding import DistilBERTEmbeddingExtractor
            cls.extractor_class = DistilBERTEmbeddingExtractor
            cls.is_available = True
        except ImportError:
            cls.is_available = False

    def setUp(self):
        if not self.is_available:
            self.skipTest("PyTorch / Transformers not installed in test environment")

    def test_model_weights_are_strictly_frozen(self):
        """Verify per GEMINI.md that DistilBERT encoder parameters do NOT require gradients."""
        extractor = self.extractor_class(max_length=64, pooling_strategy="mean")
        for name, param in extractor.model.named_parameters():
            self.assertFalse(param.requires_grad, f"Parameter {name} was not frozen!")

    def test_mean_pooling_output_shape(self):
        """Verify that mean pooling returns (N, 768) float32 embeddings."""
        extractor = self.extractor_class(max_length=64, pooling_strategy="mean")
        sample_texts = [
            "Great product, arrived quickly and works as advertised.",
            "Sobrang ganda ng item legit delivery bilis!",
        ]
        embeddings = extractor.extract(sample_texts, batch_size=2)

        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (2, 768))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_cls_pooling_output_shape(self):
        """Verify that CLS pooling returns (N, 768) float32 embeddings."""
        extractor = self.extractor_class(max_length=64, pooling_strategy="cls")
        sample_texts = ["Single review text for CLS token testing."]
        embeddings = extractor.extract(sample_texts)

        self.assertEqual(embeddings.shape, (1, 768))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_empty_input_handling(self):
        """Verify that empty list returns empty array with 768 columns."""
        extractor = self.extractor_class(max_length=64, pooling_strategy="mean")
        embeddings = extractor.extract([])
        self.assertEqual(embeddings.shape, (0, 768))


if __name__ == "__main__":
    unittest.main()
