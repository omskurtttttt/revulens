"""
RevuLens Review Text Preprocessor.

Handles text normalization, noise reduction, and character repetition reduction
for English, Filipino, and code-switched Taglish e-commerce reviews without
destroying contextual structure needed by DistilBERT.
"""

import html
import re
from typing import Dict, Any


class ReviewPreprocessor:
    """Preprocesses raw e-commerce reviews for transformer-based inference and training."""

    # Regex patterns
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    # Collapse 3 or more consecutive identical characters to at most 2 (e.g. "gandaaaaa" -> "gandaa", "legitttt" -> "legit")
    REPEATED_CHARS_PATTERN = re.compile(r"(.)\1{2,}")
    # Collapse 2 or more consecutive identical punctuation marks to 1
    REPEATED_PUNCT_PATTERN = re.compile(r"([!?,.:;])\1+")
    # Whitespace pattern
    WHITESPACE_PATTERN = re.compile(r"\s+")

    # Common Filipino / Taglish markers for heuristic language detection
    TAGLISH_MARKERS = {
        "ang", "ng", "mga", "sa", "na", "at", "po", "opo", "ko", "mo", "ni", "si",
        "ba", "pa", "din", "rin", "naman", "lang", "talaga", "sobra", "sobrang",
        "ganda", "bilis", "sulit", "ayos", "salamat", "order", "dumating", "item"
    }

    def __init__(self, min_char_length: int = 3, max_char_length: int = 2000):
        self.min_char_length = min_char_length
        self.max_char_length = max_char_length

    def clean_text(self, text: str) -> str:
        """
        Normalize and clean raw review text.
        
        Preserves casing and semantic structure for contextual embeddings,
        while stripping noise like URLs, HTML, and excessive character spam.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Decode HTML entities (e.g. &amp;, &quot;)
        cleaned = html.unescape(text)

        # 2. Strip HTML tags
        cleaned = self.HTML_TAG_PATTERN.sub(" ", cleaned)

        # 3. Strip URLs
        cleaned = self.URL_PATTERN.sub(" ", cleaned)

        # 4. Collapse repeated characters (3+ identical chars down to 2)
        cleaned = self.REPEATED_CHARS_PATTERN.sub(r"\1\1", cleaned)

        # 5. Collapse repeated punctuation
        cleaned = self.REPEATED_PUNCT_PATTERN.sub(r"\1", cleaned)

        # 6. Normalize whitespace
        cleaned = self.WHITESPACE_PATTERN.sub(" ", cleaned).strip()

        # 7. Truncate if exceeds max character threshold
        if len(cleaned) > self.max_char_length:
            cleaned = cleaned[:self.max_char_length].strip()

        return cleaned

    def is_valid_review(self, text: str) -> bool:
        """
        Check if the text has sufficient content to be meaningful for classification.
        Rejects empty strings, strings below min_char_length, or reviews without alphanumeric content.
        """
        if not text or not isinstance(text, str):
            return False

        cleaned = self.clean_text(text)
        if len(cleaned) < self.min_char_length:
            return False

        # Ensure there is at least one alphanumeric character
        if not any(c.isalnum() for c in cleaned):
            return False

        return True

    def detect_language_hint(self, text: str) -> str:
        """
        Lightweight heuristic hint to identify whether a review is English, Filipino, or Taglish.
        Used for evaluation diagnostics and logging (ISO/IEC 25010 reporting).
        """
        if not text:
            return "unknown"

        tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
        if not tokens:
            return "unknown"

        taglish_matches = tokens.intersection(self.TAGLISH_MARKERS)
        ratio = len(taglish_matches) / len(tokens)

        if len(taglish_matches) >= 2 and ratio >= 0.15:
            return "taglish" if any(t not in self.TAGLISH_MARKERS for t in tokens) else "filipino"
        elif len(taglish_matches) == 1:
            return "taglish"
        else:
            return "english"

    def process(self, raw_text: str) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline returning cleaned text and diagnostic metadata.
        """
        cleaned = self.clean_text(raw_text)
        valid = self.is_valid_review(cleaned)
        lang_hint = self.detect_language_hint(cleaned) if valid else "invalid"

        return {
            "cleaned_text": cleaned,
            "is_valid": valid,
            "char_count": len(cleaned),
            "language_hint": lang_hint,
        }
