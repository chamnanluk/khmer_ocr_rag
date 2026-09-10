from __future__ import annotations

from typing import Protocol


class Segmenter(Protocol):
    def segment(self, text: str) -> str: ...
