from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions_jsonl")
    p.add_argument("--out", default="artifacts/reports/human_annotation.csv")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rows = [json.loads(x) for x in Path(args.predictions_jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows)
    if len(df) > args.n:
        # Prefer stratification upstream. This deterministic sample is only a fallback.
        df = df.sample(args.n, random_state=args.seed)
    for col in ["annotator_correctness", "annotator_groundedness", "annotator_notes"]:
        df[col] = ""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
