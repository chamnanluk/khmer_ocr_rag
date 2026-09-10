from __future__ import annotations

import unicodedata

ZERO_WIDTH_SPACE = "\u200b"


def normalize_khmer(text: str, form: str = "NFC", preserve_boundaries: bool = True) -> str:
    """Normalize Unicode without silently inventing Khmer word boundaries.

    `preserve_boundaries=False` removes U+200B only; ordinary spaces are retained because
    Khmer spaces can have readability/functional roles and must not be rewritten blindly.
    """
    value = unicodedata.normalize(form, text)
    if not preserve_boundaries:
        value = value.replace(ZERO_WIDTH_SPACE, "")
    return value


def strip_boundary_markers(text: str) -> str:
    return text.replace(ZERO_WIDTH_SPACE, "")
