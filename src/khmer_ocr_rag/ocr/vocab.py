from __future__ import annotations

import json
from pathlib import Path

from khmer_ocr_rag.data.kcc import khmer_clusters
from khmer_ocr_rag.data.unicode import ZERO_WIDTH_SPACE

BLANK = "<blank>"


class KCCVocabulary:
    def __init__(self, tokens: list[str]):
        unique = [BLANK] + [t for t in tokens if t != BLANK]
        if ZERO_WIDTH_SPACE not in unique:
            unique.append(ZERO_WIDTH_SPACE)
        self.tokens = unique
        self.stoi = {t: i for i, t in enumerate(unique)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> KCCVocabulary:
        tokens = sorted({c for t in texts for c in khmer_clusters(t, keep_boundaries=True) if c})
        return cls(tokens)

    @property
    def blank_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [self.stoi[x] for x in khmer_clusters(text, keep_boundaries=True)]

    def decode_ctc(self, ids: list[int]) -> str:
        out = []
        prev = None
        for idx in ids:
            if idx != self.blank_id and idx != prev:
                out.append(self.tokens[idx])
            prev = idx
        return "".join(out)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.tokens, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> KCCVocabulary:
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))[1:])
