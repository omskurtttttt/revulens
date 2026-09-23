import unittest
from backend.app.constants import InternalClass, DisplayLabel, LABEL_MAPPING, get_display_label


class TestConstants(unittest.TestCase):
    def test_label_mapping_integrity(self):
        """Ensure all internal model classes map to defined display labels."""
        self.assertEqual(LABEL_MAPPING[InternalClass.GENUINE], DisplayLabel.LIKELY_GENUINE)
        self.assertEqual(LABEL_MAPPING[InternalClass.DECEPTIVE], DisplayLabel.POTENTIALLY_DECEPTIVE)

    def test_get_display_label(self):
        """Ensure conversion function outputs correct display labels and handles fallbacks."""
        self.assertEqual(get_display_label("Genuine"), "Likely Genuine")
        self.assertEqual(get_display_label("Deceptive"), "Potentially Deceptive")
        # Unrecognized string should fall back gracefully
        self.assertEqual(get_display_label("UnknownClass"), "UnknownClass")


if __name__ == "__main__":
    unittest.main()
