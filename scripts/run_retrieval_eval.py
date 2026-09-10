from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from khmer_ocr_rag.experiments.retrieval_eval import evaluate_retriever
from khmer_ocr_rag.metrics.retrieval import aggregate_query_metrics
from khmer_ocr_rag.retrieval.bm25 import BM25Retriever


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_retriever(name: str, model_name: str | None = None):
    if name == "bm25":
        return BM25Retriever()
    if name == "dense":
        from khmer_ocr_rag.retrieval.dense import DenseRetriever
        return DenseRetriever(model_name or "BAAI/bge-m3")
    if name == "hybrid":
        from khmer_ocr_rag.retrieval.dense import DenseRetriever
        from khmer_ocr_rag.retrieval.hybrid import ReciprocalRankFusionRetriever
        return ReciprocalRankFusionRetriever([BM25Retriever(), DenseRetriever(model_name or "BAAI/bge-m3")])
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--passages", required=True)
    p.add_argument("--qa", required=True)
    p.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
    p.add_argument("--dense-model", default="BAAI/bge-m3")
    p.add_argument("--out", default="artifacts/results/retrieval_eval.csv")
    args = p.parse_args()
    passages = load_jsonl(args.passages)
    qa = load_jsonl(args.qa)
    retriever = build_retriever(args.retriever, args.dense_model).fit(passages)
    rows = evaluate_retriever(retriever, qa)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(aggregate_query_metrics(rows))
    print(out)


if __name__ == "__main__":
    main()
