from khmer_ocr_rag.retrieval.bm25 import BM25Retriever


def test_bm25_exact_khmer_term():
    docs = [
        {"passage_id": "p1", "text_condition": "រាជធានី ភ្នំពេញ"},
        {"passage_id": "p2", "text_condition": "ខេត្ត សៀមរាប"},
    ]
    hits = BM25Retriever().fit(docs).search("ភ្នំពេញ", 2)
    assert hits[0].passage_id == "p1"
