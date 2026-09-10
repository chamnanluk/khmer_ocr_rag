from __future__ import annotations

from khmer_ocr_rag.data.unicode import ZERO_WIDTH_SPACE


class WhitespaceBoundarySegmenter:
    """Sanity-check baseline only; not a publishable Khmer segmenter."""

    def segment(self, text: str) -> str:
        return ZERO_WIDTH_SPACE.join(text.split())
