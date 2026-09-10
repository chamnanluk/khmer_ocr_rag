from __future__ import annotations

"""Materialize the proposal's P0–P5 text conditions from aligned records.

Expected fields per passage row:
- text_gold
- text_gold_segmented
- text_ocr
- text_sequential_segmented
- text_joint_segmented
- text_ocr_gold_segmented (optional P5 if an oracle boundary projection exists)

P4 is gold text with the segmentation pattern predicted by the selected segmenter. Creating
P4 requires a segmentation prediction on gold text; it must not reuse the OCR-corrupted text.
"""

from khmer_ocr_rag.data.unicode import ZERO_WIDTH_SPACE, strip_boundary_markers


def materialize_condition(row: dict, condition: str) -> str:
    condition = condition.upper()
    if condition == "P0":
        return row["text_gold_segmented"]
    if condition == "P1":
        return strip_boundary_markers(row["text_ocr"])
    if condition == "P2":
        return row["text_sequential_segmented"]
    if condition == "P3":
        return row["text_joint_segmented"]
    if condition == "P4":
        return row["text_gold_with_predicted_boundaries"]
    if condition == "P5":
        return row["text_ocr_with_gold_boundaries"]
    raise ValueError(f"Unknown condition: {condition}")


def project_boundaries(source_segmented: str, target_unsegmented: str) -> str:
    """Project boundary offsets only when source and target underlying text are identical.

    For imperfect OCR, gold-boundary projection requires a documented sequence alignment and
    cannot be done safely with raw offsets. This helper intentionally refuses that case.
    """
    source_plain = strip_boundary_markers(source_segmented)
    if source_plain != target_unsegmented:
        raise ValueError("Direct boundary projection requires identical underlying text")
    offsets = []
    plain_i = 0
    for ch in source_segmented:
        if ch == ZERO_WIDTH_SPACE:
            offsets.append(plain_i)
        else:
            plain_i += 1
    out = list(target_unsegmented)
    for pos in reversed(offsets):
        out.insert(pos, ZERO_WIDTH_SPACE)
    return "".join(out)
