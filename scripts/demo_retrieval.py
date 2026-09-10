from __future__ import annotations

import json
from pathlib import Path

from khmer_ocr_rag.experiments.retrieval_eval import evaluate_retriever
from khmer_ocr_rag.metrics.retrieval import aggregate_query_metrics
from khmer_ocr_rag.retrieval.bm25 import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    passages = load_jsonl(ROOT / "data/annotations/example_passages.jsonl")
    qa = load_jsonl(ROOT / "data/annotations/example_qa.jsonl")
    retriever = BM25Retriever().fit(passages)
    rows = evaluate_retriever(retriever, qa, k_values=(1, 3))
    for row in rows:
        print(row)
    print("mean:", aggregate_query_metrics(rows))


if __name__ == "__main__":
    main()
