"""
Unit Tests for Extension Contracts, Manifest V3, and Pre-Deployment Verification (Step 8).

Strictly verifies GEMINI.md rules:
1. Manifest V3 compliance and background service worker declaration.
2. Content scripts NEVER call the backend directly (no direct fetch in content_script.js).
3. Selection capture via window.getSelection() without fragile per-platform DOM parsing.
4. Exact label parity with backend/app/constants.py ("Likely Genuine" / "Potentially Deceptive").
5. Color mapping and disclaimer text compliance (no claims of "verified authentic" or "fake").
"""

import os
import re
import json
import unittest

from backend.app.constants import InternalClass, DisplayLabel, LABEL_MAPPING


class TestExtensionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ext_dir = "extension"
        cls.manifest_path = os.path.join(cls.ext_dir, "manifest.json")
        cls.config_path = os.path.join(cls.ext_dir, "config.js")
        cls.worker_path = os.path.join(cls.ext_dir, "background", "service_worker.js")
        cls.content_path = os.path.join(cls.ext_dir, "content", "content_script.js")
        cls.css_path = os.path.join(cls.ext_dir, "content", "overlay.css")

    def test_manifest_v3_structure(self):
        """Verify manifest.json is valid Manifest V3 with service worker and permissions."""
        self.assertTrue(os.path.exists(self.manifest_path), "manifest.json must exist")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest.get("manifest_version"), 3)
        self.assertIn("background", manifest)
        self.assertEqual(manifest["background"].get("service_worker"), "background/service_worker.js")
        self.assertTrue(os.path.exists(self.worker_path), "Declared service_worker file must exist")

        # Verify host permissions include localhost backend
        host_perms = manifest.get("host_permissions", [])
        has_backend_perm = any("localhost" in p or "127.0.0.1" in p for p in host_perms)
        self.assertTrue(has_backend_perm, "Host permissions must allow backend origin")

        # Verify content scripts load config and content script
        content_scripts = manifest.get("content_scripts", [])
        self.assertGreater(len(content_scripts), 0)
        js_files = content_scripts[0].get("js", [])
        self.assertIn("config.js", js_files)
        self.assertIn("content/content_script.js", js_files)

    def test_content_script_mixed_content_safety(self):
        """
        Verify GEMINI.md rule: Do not call the backend from the content script.
        On https pages, browsers block calls to an http localhost backend.
        Route requests through the service worker.
        """
        self.assertTrue(os.path.exists(self.content_path), "content_script.js must exist")
        with open(self.content_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Content script must use chrome.runtime.sendMessage
        self.assertIn("chrome.runtime.sendMessage", content)
        # Content script must NOT call fetch() directly
        fetch_matches = re.findall(r"\bfetch\s*\(", content)
        self.assertEqual(len(fetch_matches), 0, "content_script.js must not contain direct fetch() calls")

    def test_selection_capture_without_platform_selectors(self):
        """
        Verify GEMINI.md rule: The consumer selects (highlights) review text (window.getSelection()).
        Do not write platform-specific DOM parsing.
        """
        with open(self.content_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("window.getSelection()", content)
        # Must not contain platform-specific class queries like shopee review selectors
        self.assertNotIn(".shopee-product-rating", content)
        self.assertNotIn("lazada", content.lower())

    def test_config_label_and_color_parity(self):
        """
        Verify config.js label mapping matches backend constants.py and defines color map.
        """
        self.assertTrue(os.path.exists(self.config_path), "config.js must exist")
        with open(self.config_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Labels parity
        self.assertTrue(
            f'"{InternalClass.GENUINE.value}": "{DisplayLabel.LIKELY_GENUINE.value}"' in content or
            f'{InternalClass.GENUINE.value}: "{DisplayLabel.LIKELY_GENUINE.value}"' in content
        )
        self.assertTrue(
            f'"{InternalClass.DECEPTIVE.value}": "{DisplayLabel.POTENTIALLY_DECEPTIVE.value}"' in content or
            f'{InternalClass.DECEPTIVE.value}: "{DisplayLabel.POTENTIALLY_DECEPTIVE.value}"' in content
        )

        # Color map presence
        self.assertIn("COLOR_MAP", content)
        self.assertIn("deceptive", content)
        self.assertIn("genuine", content)

        # Disclaimer presence
        self.assertIn("DISCLAIMER_TEXT", content)

    def test_word_labeling_terminology_guardrail(self):
        """
        Verify GEMINI.md rule: Do not label words as 'fake'.
        Never write UI copy claiming Genuine is verified authentic or Deceptive is human-written fake.
        """
        for filepath in [self.content_path, self.config_path, self.css_path]:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Check for prohibited claims
            self.assertNotIn("fake word", content.lower())
            self.assertNotIn("verified authentic", content.lower())

    def test_no_confidence_meter_in_ui(self):
        """
        Verify GEMINI.md rule: No confidence score anywhere (UI, API, logs).
        content_script.js must not contain a confidence meter or display confidence percentages.
        """
        with open(self.content_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("revulens-confidence-container", content)
        self.assertNotIn("confidence * 100", content)

    def test_inline_word_highlights_implementation(self):
        """
        Verify GEMINI.md TODO 3 inline highlighting requirements:
        1. applyInlineWordHighlights is implemented in content_script.js.
        2. overlay.css defines .revulens-inline-highlight, .revulens-inline-deceptive, .revulens-inline-genuine.
        3. Deceptive-leaning has solid underline; Genuine-leaning has dotted underline (accessible).
        4. Tooltips say 'toward Deceptive' or 'toward Genuine', never numbers.
        5. Terminology uses 'words that influenced the result', never 'fake words' or 'deceptive words'.
        6. Potentially Deceptive badge uses amber styling per GEMINI.md.
        """
        with open(self.content_path, "r", encoding="utf-8") as f:
            content = f.read()
        with open(self.css_path, "r", encoding="utf-8") as f:
            css = f.read()

        # 1. Implementation presence
        self.assertIn("function applyInlineWordHighlights(", content)
        self.assertIn("clearWordHighlights()", content)

        # 2. CSS classes
        self.assertIn(".revulens-inline-highlight", css)
        self.assertIn(".revulens-inline-deceptive", css)
        self.assertIn(".revulens-inline-genuine", css)

        # 3. Accessibility underlines
        self.assertIn("solid", css)
        self.assertIn("dotted", css)

        # 4. Tooltip wording
        self.assertIn('"toward Deceptive"', content)
        self.assertIn('"toward Genuine"', content)

        # 5. Terminology
        self.assertIn("Words that influenced the result", content)
        self.assertNotIn("deceptive words", content.lower())
        self.assertNotIn("fake words", content.lower())

        # 6. Amber badge styling for deceptive
        self.assertIn("#fffbeb", css)

    def test_popup_gemini_compliance(self):
        """
        Verify GEMINI.md popup rules:
        - Minimal: on/off toggle (make it work or remove it), 'How to use'
          (including that Taglish behavior is exploratory), and a short explanation
          of the highlight colors. Nothing else.
        - Do not build: Confidence meter, automatic page scanning, full-screen modals,
          per-platform DOM parsing, user accounts, history.
        """
        popup_html_path = os.path.join(self.ext_dir, "popup", "popup.html")
        popup_js_path = os.path.join(self.ext_dir, "popup", "popup.js")

        self.assertTrue(os.path.exists(popup_html_path))
        self.assertTrue(os.path.exists(popup_js_path))

        with open(popup_html_path, "r", encoding="utf-8") as f:
            html = f.read()
        with open(popup_js_path, "r", encoding="utf-8") as f:
            js = f.read()

        # On/off toggle exists and binds to revulensEnabled
        self.assertIn('id="toggle-revulens"', html)
        self.assertIn("revulensEnabled", js)

        # How to use section present and mentions Taglish is exploratory
        self.assertIn("How to Use", html)
        self.assertIn("Taglish", html)
        self.assertIn("exploratory", html.lower())

        # Highlight colors explained
        self.assertIn("Highlight Colors", html)
        self.assertIn("solid underline", html)
        self.assertIn("dotted underline", html)

        # Strict prohibition: No scan stats, history, or counters
        self.assertNotIn("total-analyzed", html)
        self.assertNotIn("count-genuine", html)
        self.assertNotIn("count-deceptive", html)
        self.assertNotIn("totalScanned", js)


if __name__ == "__main__":
    unittest.main()
