"""
RevuLens Centralized Constants & Label Mappings.

All label mappings between internal model classes and user-facing UI labels
must be defined here. Do not hardcode these labels elsewhere.
"""

from enum import Enum
from typing import Dict


class InternalClass(str, Enum):
    GENUINE = "Genuine"
    DECEPTIVE = "Deceptive"


class DisplayLabel(str, Enum):
    LIKELY_GENUINE = "Likely Genuine"
    POTENTIALLY_DECEPTIVE = "Potentially Deceptive"


# Central mapping from model prediction to user-facing display label
LABEL_MAPPING: Dict[InternalClass, DisplayLabel] = {
    InternalClass.GENUINE: DisplayLabel.LIKELY_GENUINE,
    InternalClass.DECEPTIVE: DisplayLabel.POTENTIALLY_DECEPTIVE,
}


def get_display_label(internal_class: str) -> str:
    """Safely convert an internal model class string to its user-facing label."""
    try:
        class_enum = InternalClass(internal_class)
        return LABEL_MAPPING[class_enum].value
    except (ValueError, KeyError):
        return internal_class
