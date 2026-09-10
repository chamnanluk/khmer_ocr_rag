from __future__ import annotations

from dataclasses import dataclass

from khmer_ocr_rag.data.kcc import khmer_clusters
from khmer_ocr_rag.data.unicode import ZERO_WIDTH_SPACE


@dataclass(frozen=True)
class SegmentationExample:
    clusters: list[str]
    labels: list[int]


def segmented_to_example(text: str) -> SegmentationExample:
    """Convert U+200B segmented text to KCC sequence + boundary-after labels.

    labels[i] denotes whether a word boundary follows cluster i. The final cluster label
    is kept as 0 and ignored by training/evaluation if desired.
    """
    items = khmer_clusters(text, keep_boundaries=True)
    clusters: list[str] = []
    labels: list[int] = []
    for item in items:
        if item == ZERO_WIDTH_SPACE:
            if labels:
                labels[-1] = 1
            continue
        clusters.append(item)
        labels.append(0)
    return SegmentationExample(clusters, labels)


class BoundaryDataset:
    def __init__(self, segmented_texts: list[str], vocab):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install OCR extras: pip install -e '.[ocr]'") from exc
        self.torch = torch
        self.examples = [segmented_to_example(t) for t in segmented_texts]
        self.vocab = vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        ids = [self.vocab.stoi[c] for c in ex.clusters]
        return {
            "ids": self.torch.tensor(ids, dtype=self.torch.long),
            "labels": self.torch.tensor(ex.labels, dtype=self.torch.float32),
        }


def collate_boundaries(batch):
    import torch
    max_len = max(len(x["ids"]) for x in batch)
    ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros((len(batch), max_len), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = len(item["ids"])
        ids[i, :n] = item["ids"]
        labels[i, :n] = item["labels"]
        mask[i, :n] = True
    return {"ids": ids, "labels": labels, "mask": mask}
