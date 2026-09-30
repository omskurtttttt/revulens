"""
Tests for shared preprocessing function parity between training and inference paths.
"""

import unittest
from backend.app.preprocessing import (
    normalize_text,
    preprocess_for_training,
    preprocess_for_inference,
)


class TestPreprocessingParity(unittest.TestCase):
    def test_training_and_inference_parity(self):
        """Verify that training and inference paths give 100% identical output."""
        sample_texts = [
            "Love this! Well made, sturdy, and very comfortable. I love it! Very pretty.",
            "Super ganda ng item legit delivery bilis! Salamat sa seller.",
            "<p>Check this out: https://shopee.ph/product-123 &quot;Authentic&quot;!</p>",
            "   Multiple   spaces \n and \t tabs...   ",
            "This isn't just 'good' — it's the best item I’ve ever bought!",
        ]
        for text in sample_texts:
            train_output = preprocess_for_training(text)
            infer_output = preprocess_for_inference(text)
            self.assertEqual(train_output, infer_output)
            self.assertEqual(train_output, normalize_text(text))

    def test_normalization_steps(self):
        """Verify exact preprocessing transformations per GEMINI.md."""
        raw = "<p>Check https://shopee.ph/item ! <b>Great Quality</b>, fast delivery... Highly recommend!!</p>"
        expected = "check great quality fast delivery highly recommend"
        self.assertEqual(normalize_text(raw), expected)

    def test_no_stopword_removal(self):
        """Verify that stop words are preserved for contextual embeddings."""
        text = "This is a great product for the price and it works well with my laptop."
        normalized = normalize_text(text)
        # Ensure stop words are NOT stripped
        for sw in ["this", "is", "a", "for", "the", "and", "it", "with", "my"]:
            self.assertIn(sw, normalized.split())

    def test_taglish_normalization_matches_firecs_format(self):
        """Verify that Taglish text with punctuation/uppercase normalizes to clean lowercase."""
        raw = "Sobrang GANDA ng item! Dumating kahapon, legit talaga... Salamat seller & rider!"
        expected = "sobrang ganda ng item dumating kahapon legit talaga salamat seller rider"
        self.assertEqual(normalize_text(raw), expected)

    def test_empty_and_edge_inputs(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")
        self.assertEqual(normalize_text("    "), "")
        self.assertEqual(normalize_text("!@#$%^&*()"), "")


if __name__ == "__main__":
    unittest.main()
