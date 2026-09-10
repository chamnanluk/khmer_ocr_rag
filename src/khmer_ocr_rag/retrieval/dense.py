from __future__ import annotations

import numpy as np

from .base import SearchHit


class DenseRetriever:
    """Sentence-transformers adapter with exact cosine search for reproducibility."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install dense extras: pip install -e '.[dense]'") from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.passages: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def fit(self, passages: list[dict]) -> DenseRetriever:
        self.passages = passages
        texts = [p["text_condition"] for p in passages]
        self.embeddings = np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
        return self

    def search(self, query: str, k: int = 10) -> list[SearchHit]:
        if self.embeddings is None:
            raise RuntimeError("Call fit() first")
        q = np.asarray(self.model.encode([query], normalize_embeddings=True, show_progress_bar=False))[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)[:k]
        return [
            SearchHit(self.passages[i]["passage_id"], float(scores[i]), self.passages[i]["text_condition"], rank)
            for rank, i in enumerate(order, 1)
        ]
