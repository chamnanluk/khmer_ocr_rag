from __future__ import annotations

from khmer_ocr_rag.data.unicode import normalize_khmer


def normalized_exact_match(reference: str, hypothesis: str) -> float:
    return float(normalize_khmer(reference).strip() == normalize_khmer(hypothesis).strip())


def answer_contains(reference: str, hypothesis: str) -> float:
    return float(normalize_khmer(reference).strip() in normalize_khmer(hypothesis))


def evidence_recall(retrieved_ids: list[str], supporting_ids: set[str]) -> float:
    if not supporting_ids:
        return 0.0
    return len(set(retrieved_ids) & supporting_ids) / len(supporting_ids)
