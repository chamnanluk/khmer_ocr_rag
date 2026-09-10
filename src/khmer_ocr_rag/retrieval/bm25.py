from __future__ import annotations

import math
import re
from collections import Counter

from .base import SearchHit


def _tokens(text: str) -> list[str]:
    # Default sparse baseline is deliberately Unicode-safe and configurable later.
    # If explicit U+200B boundaries exist, use them as separators; otherwise fall back
    # to whitespace and character n-grams for Khmer continuous text.
    text = text.replace("\u200b", " ").strip()
    chunks = [x for x in re.split(r"\s+", text) if x]
    if len(chunks) > 1:
        return chunks
    compact = "".join(chunks)
    if not compact:
        return []
    return [compact[i:i+2] for i in range(max(1, len(compact)-1))]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.passages: list[dict] = []
        self.docs: list[list[str]] = []
        self.df: Counter[str] = Counter()
        self.avgdl = 0.0

    def fit(self, passages: list[dict]) -> BM25Retriever:
        self.passages = passages
        self.docs = [_tokens(p["text_condition"]) for p in passages]
        self.df = Counter()
        for doc in self.docs:
            self.df.update(set(doc))
        self.avgdl = sum(map(len, self.docs)) / len(self.docs) if self.docs else 0.0
        return self

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if n else 0.0

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        q = _tokens(query)
        scored = []
        for i, doc in enumerate(self.docs):
            tf = Counter(doc)
            dl = len(doc)
            score = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                score += self._idf(term) * (f * (self.k1 + 1)) / denom
            scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], self.passages[x[1]]["passage_id"]))
        hits = []
        for rank, (score, i) in enumerate(scored[:k], 1):
            p = self.passages[i]
            hits.append(SearchHit(p["passage_id"], float(score), p["text_condition"], rank))
        return hits
