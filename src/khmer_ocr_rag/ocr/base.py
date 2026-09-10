from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class OCRPrediction:
    sample_id: str
    text: str
    latency_ms: float | None = None
    metadata: dict | None = None


class OCRRecognizer(Protocol):
    def predict(self, image, *, emit_boundaries: bool = False) -> str: ...
