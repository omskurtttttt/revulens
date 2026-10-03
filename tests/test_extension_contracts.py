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


if __name__ == "__main__":
    unittest.main()
