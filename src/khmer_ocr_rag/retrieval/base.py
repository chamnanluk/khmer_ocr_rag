from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchHit:
    passage_id: str
    score: float
    text: str
    rank: int


class Retriever(Protocol):
    def fit(self, passages: list[dict]) -> Retriever: ...
    def search(self, query: str, k: int = 10) -> list[SearchHit]: ...
