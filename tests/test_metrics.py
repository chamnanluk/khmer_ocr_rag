from khmer_ocr_rag.metrics.edit import levenshtein
from khmer_ocr_rag.metrics.ocr import cer
from khmer_ocr_rag.metrics.retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from khmer_ocr_rag.metrics.segmentation import boundary_f1


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3


def test_cer_ignores_boundary_marker():
    assert cer("abc\u200bdef", "abcdef") == 0.0


def test_boundary_f1():
    score = boundary_f1("ab\u200bcd\u200bef", "ab\u200bcd\u200bef")
    assert score.f1 == 1.0
    score2 = boundary_f1("ab\u200bcd\u200bef", "abcd\u200bef")
    assert 0 < score2.f1 < 1


def test_retrieval_metrics():
    ranked = ["a", "b", "c"]
    rel = {"b"}
    assert recall_at_k(ranked, rel, 1) == 0.0
    assert recall_at_k(ranked, rel, 2) == 1.0
    assert reciprocal_rank(ranked, rel) == 0.5
    assert 0 < ndcg_at_k(ranked, rel, 3) < 1
