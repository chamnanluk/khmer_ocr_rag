from __future__ import annotations

from khmer_ocr_rag.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def evaluate_retriever(retriever, qa_rows: list[dict], k_values=(1, 3, 5, 10)) -> list[dict]:
    rows = []
    max_k = max(k_values)
    for q in qa_rows:
        hits = retriever.search(q.get("question_for_retrieval", q["question_km"]), max_k)
        ids = [h.passage_id for h in hits]
        relevant = set(q["relevant_passage_ids"])
        row = {"query_id": q["query_id"], "mrr": reciprocal_rank(ids, relevant, max_k)}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(ids, relevant, k)
            row[f"precision@{k}"] = precision_at_k(ids, relevant, k)
            row[f"hit@{k}"] = hit_rate_at_k(ids, relevant, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ids, relevant, k)
        rows.append(row)
    return rows
