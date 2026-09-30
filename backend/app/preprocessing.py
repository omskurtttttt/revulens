"""
RevuLens Shared Text Preprocessing Module.

Single source of truth for text normalization used across both training notebooks
and real-time inference (FastAPI backend & explainability engine).

Standardized steps per thesis requirements (GEMINI.md):
1. Decode HTML entities and strip HTML tags
2. Remove URLs
3. Convert to lowercase
4. Remove punctuation (matches FiReCS formatting to prevent source/label leakage)
5. Collapse redundant whitespace
6. NO stop-word removal (preserves contextual structure for DistilBERT)
"""

import html
import re
import string

# Precompiled regex patterns for maximum throughput
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")

# Comprehensive punctuation pattern including ASCII and unicode punctuation
PUNCTUATION_RE = re.compile(r"[\s" + re.escape(string.punctuation) + r"“”‘’—–…•·\t\r\n]+")


def normalize_text(text: str) -> str:
    """
    Standard text normalization pipeline for both training and inference.
    
    Returns clean, lowercase, punctuation-free text with collapsed whitespace.
    Does NOT remove stop words to preserve transformer attention representations.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Decode HTML entities (e.g., &amp; -> &) and remove HTML tags
    cleaned = html.unescape(text)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)

    # 2. Remove URLs
    cleaned = URL_RE.sub(" ", cleaned)

    # 3. Lowercase
    cleaned = cleaned.lower()

    # 4. Remove punctuation (replace with single space to prevent fusing adjacent words)
    cleaned = PUNCTUATION_RE.sub(" ", cleaned)

    # 5. Collapse multiple whitespaces and strip leading/trailing spaces
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()

    return cleaned


def preprocess_for_training(text: str) -> str:
    """Entry point for training data preparation notebooks."""
    return normalize_text(text)


def preprocess_for_inference(text: str) -> str:
    """Entry point for FastAPI backend inference."""
    return normalize_text(text)
