from __future__ import annotations

from khmer_ocr_rag.data.kcc import khmer_clusters
from khmer_ocr_rag.data.unicode import normalize_khmer, strip_boundary_markers

from .edit import error_rate


def cer(reference: str, hypothesis: str, normalize: bool = False) -> float:
    ref = strip_boundary_markers(reference)
    hyp = strip_boundary_markers(hypothesis)
    if normalize:
        ref, hyp = normalize_khmer(ref), normalize_khmer(hyp)
    return error_rate(list(ref), list(hyp))


def kcc_error_rate(reference: str, hypothesis: str, normalize: bool = True) -> float:
    ref = strip_boundary_markers(reference)
    hyp = strip_boundary_markers(hypothesis)
    if normalize:
        ref, hyp = normalize_khmer(ref), normalize_khmer(hyp)
    return error_rate(khmer_clusters(ref, keep_boundaries=False), khmer_clusters(hyp, keep_boundaries=False))
