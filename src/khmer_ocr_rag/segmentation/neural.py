from __future__ import annotations

from dataclasses import dataclass

from khmer_ocr_rag.data.kcc import join_clusters
from khmer_ocr_rag.data.unicode import ZERO_WIDTH_SPACE


@dataclass
class BoundaryPrediction:
    segmented_text: str
    boundary_probabilities: list[float]


class NeuralBoundarySegmenter:
    """Trainable KCC-level boundary classifier used for P2.

    The network is loaded lazily so the base package remains lightweight.
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 256):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("Install OCR extras: pip install -e '.[ocr]'") from exc
        self.torch = torch
        self.nn = nn
        self.model = nn.ModuleDict({
            "embedding": nn.Embedding(vocab_size, hidden_dim),
            "encoder": nn.LSTM(hidden_dim, hidden_dim // 2, bidirectional=True, batch_first=True),
            "head": nn.Linear(hidden_dim, 1),
        })

    def logits(self, token_ids):
        x = self.model["embedding"](token_ids)
        x, _ = self.model["encoder"](x)
        return self.model["head"](x).squeeze(-1)

    @staticmethod
    def decode(clusters: list[str], probabilities: list[float], threshold: float = 0.5) -> BoundaryPrediction:
        if len(probabilities) != max(0, len(clusters) - 1):
            raise ValueError("Need one boundary probability between each adjacent cluster")
        out = []
        for i, cluster in enumerate(clusters):
            out.append(cluster)
            if i < len(probabilities) and probabilities[i] >= threshold:
                out.append(ZERO_WIDTH_SPACE)
        return BoundaryPrediction(join_clusters(out), probabilities)
