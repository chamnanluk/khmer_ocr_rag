from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_run_artifacts(out_dir: str | Path, metrics: list[dict], predictions: list[dict], provenance: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
