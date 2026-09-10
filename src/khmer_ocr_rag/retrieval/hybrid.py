from __future__ import annotations

from collections import defaultdict

from .base import SearchHit


class ReciprocalRankFusionRetriever:
    def __init__(self, retrievers: list, rrf_k: int = 60, candidate_k: int = 100):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def fit(self, passages: list[dict]):
        for retriever in self.retrievers:
            retriever.fit(passages)
        return self

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        scores: dict[str, float] = defaultdict(float)
        texts: dict[str, str] = {}
        for retriever in self.retrievers:
            for hit in retriever.search(query, self.candidate_k):
                scores[hit.passage_id] += 1.0 / (self.rrf_k + hit.rank)
                texts[hit.passage_id] = hit.text
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [SearchHit(pid, score, texts[pid], rank) for rank, (pid, score) in enumerate(ranked, 1)]
