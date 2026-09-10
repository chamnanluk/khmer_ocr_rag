from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from khmer_ocr_rag.experiments.controlled_retrieval import run_bm25_rate_curve
from khmer_ocr_rag.perturbations.khmer_errors import ErrorFamily

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    passages = load(ROOT / "data/annotations/example_passages.jsonl")
    qa = load(ROOT / "data/annotations/example_qa.jsonl")
    rows = []
    for family in (ErrorFamily.BASE, ErrorFamily.BOUNDARY_INSERT, ErrorFamily.BOUNDARY_DELETE):
        rows.extend(run_bm25_rate_curve(passages, qa, family, [0.0, 0.05, 0.10], seed=42))
    out = ROOT / "artifacts/results/example_e3.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
