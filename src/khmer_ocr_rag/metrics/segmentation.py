from __future__ import annotations

from dataclasses import dataclass

from khmer_ocr_rag.data.unicode import ZERO_WIDTH_SPACE, strip_boundary_markers


@dataclass(frozen=True)
class BoundaryScores:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def boundary_positions(segmented: str) -> set[int]:
    positions: set[int] = set()
    visible_index = 0
    for ch in segmented:
        if ch == ZERO_WIDTH_SPACE:
            positions.add(visible_index)
        else:
            visible_index += 1
    return positions


def boundary_f1(reference_segmented: str, hypothesis_segmented: str) -> BoundaryScores:
    # If the underlying character strings differ, alignable boundary evaluation is ambiguous.
    # For RQ1 report this score on exact/normalized-matching transcription subsets or after a
    # documented alignment procedure. Here we require identical underlying strings.
    if strip_boundary_markers(reference_segmented) != strip_boundary_markers(hypothesis_segmented):
        raise ValueError("Boundary F1 requires identical underlying text in this baseline implementation")
    gold = boundary_positions(reference_segmented)
    pred = boundary_positions(hypothesis_segmented)
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 1.0 if not gold else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BoundaryScores(precision, recall, f1, tp, fp, fn)
