import numpy as np

from khmer_ocr_rag.perturbations.khmer_errors import ErrorFamily, corrupt_text


def test_boundary_delete_is_deterministic():
    text = "កម្ពុជា\u200bមាន\u200bភ្នំពេញ"
    a = corrupt_text(text, ErrorFamily.BOUNDARY_DELETE, 0.5, np.random.default_rng(7))
    b = corrupt_text(text, ErrorFamily.BOUNDARY_DELETE, 0.5, np.random.default_rng(7))
    assert a == b
    assert len(a.text) <= len(text)


def test_boundary_insert_adds_marker():
    text = "កម្ពុជាមានភ្នំពេញ"
    result = corrupt_text(text, ErrorFamily.BOUNDARY_INSERT, 0.1, np.random.default_rng(1))
    assert result.text.count("\u200b") >= 1


def test_subscript_corruption_targets_coeng_unit():
    text = "ក្រ"  # KA + COENG + RO in Unicode Khmer spelling
    result = corrupt_text(text, ErrorFamily.SUBSCRIPT, 1.0, np.random.default_rng(0))
    assert result.text == "ក"
