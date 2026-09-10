from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    retrieved_passage_ids: list[str]
    context: list[str]
    metadata: dict


class Generator(Protocol):
    def generate(self, question: str, contexts: list[str]) -> str: ...
