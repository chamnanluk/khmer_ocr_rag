from __future__ import annotations

from copy import deepcopy

import numpy as np

from khmer_ocr_rag.experiments.retrieval_eval import evaluate_retriever
from khmer_ocr_rag.metrics.retrieval import aggregate_query_metrics
from khmer_ocr_rag.perturbations.khmer_errors import ErrorFamily, corrupt_text
from khmer_ocr_rag.retrieval.bm25 import BM25Retriever


def run_bm25_rate_curve(
    passages: list[dict],
    qa_rows: list[dict],
    family: ErrorFamily,
    rates: list[float],
    seed: int = 42,
) -> list[dict]:
    results = []
    for rate in rates:
        rng = np.random.default_rng(seed)
        variant = deepcopy(passages)
        edit_count = 0
        realized = []
        for p in variant:
            source = p.get("text_segmented", p.get("text_condition", p["text_gold"]))
            r = corrupt_text(source, family, rate, rng)
            p["text_condition"] = r.text
            p["condition"] = f"{family.value}_{rate:.3f}"
            edit_count += len(r.edits)
            realized.append(r.realized_rate)
        retriever = BM25Retriever().fit(variant)
        per_query = evaluate_retriever(retriever, qa_rows, k_values=(1, 3, 5, 10))
        agg = aggregate_query_metrics(per_query)
        results.append({
            "error_family": family.value,
            "target_rate": rate,
            "mean_realized_rate": float(np.mean(realized)) if realized else 0.0,
            "edits": edit_count,
            **agg,
        })
    return results
