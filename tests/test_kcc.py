from khmer_ocr_rag.data.kcc import khmer_clusters
from khmer_ocr_rag.data.unicode import ZERO_WIDTH_SPACE


def test_boundary_preserved():
    items = khmer_clusters("កម្ពុជា" + ZERO_WIDTH_SPACE + "មាន")
    assert ZERO_WIDTH_SPACE in items
    assert "".join(items) == "កម្ពុជា" + ZERO_WIDTH_SPACE + "មាន"
