import unittest
from backend.app.services.preprocessor import ReviewPreprocessor


class TestReviewPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ReviewPreprocessor(min_char_length=3, max_char_length=500)

    def test_html_and_url_removal(self):
        raw = "Check this <a href='https://shopee.ph/item'>link</a>! It's &amp; authentic!"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("<a", cleaned)
        self.assertIn("&", cleaned)
        self.assertIn("authentic!", cleaned)

    def test_character_repetition_collapse(self):
        # 3+ identical chars should be collapsed to at most 2 (preserving valid double letters like 'oo', 'll')
        raw = "Gandaaaaa sobraaaa! Legitttttt item!"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertEqual(cleaned, "Gandaa sobraa! Legitt item!")

    def test_punctuation_repetition_collapse(self):
        raw = "Super fast delivery!!!!! Authentic????"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertEqual(cleaned, "Super fast delivery! Authentic?")

    def test_whitespace_normalization(self):
        raw = "  Order   received    in good \n condition. \t Thank you!   "
        cleaned = self.preprocessor.clean_text(raw)
        self.assertEqual(cleaned, "Order received in good condition. Thank you!")

    def test_valid_review_criteria(self):
        self.assertTrue(self.preprocessor.is_valid_review("Maganda ang item"))
        self.assertTrue(self.preprocessor.is_valid_review("Good quality"))
        
        # Invalid cases
        self.assertFalse(self.preprocessor.is_valid_review(""))
        self.assertFalse(self.preprocessor.is_valid_review("   "))
        self.assertFalse(self.preprocessor.is_valid_review(".."))
        self.assertFalse(self.preprocessor.is_valid_review("🔥👍"))

    def test_language_hint_detection(self):
        english_text = "The product arrived on time and the build quality is great."
        taglish_text = "Sobrang bilis dumating ng order ko, legit ang item salamat seller!"
        
        self.assertEqual(self.preprocessor.detect_language_hint(english_text), "english")
        self.assertIn(self.preprocessor.detect_language_hint(taglish_text), ["taglish", "filipino"])

    def test_process_pipeline_output(self):
        result = self.preprocessor.process("Sobrang gandaaaa ng packaging! https://shopee.ph")
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["cleaned_text"], "Sobrang gandaa ng packaging!")
        self.assertGreater(result["char_count"], 0)
        self.assertIn(result["language_hint"], ["taglish", "filipino"])


if __name__ == "__main__":
    unittest.main()
