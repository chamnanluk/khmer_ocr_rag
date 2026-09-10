from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd

from khmer_ocr_rag.utils.repro import config_hash, load_yaml, set_seed


def run_config(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    cfg = load_yaml(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    run_id = f"{cfg.get('experiment','run')}_{time.strftime('%Y%m%d-%H%M%S')}_{config_hash(cfg)}"
    out = Path(cfg.get("output_root", "artifacts/results")) / run_id
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, out / "config.yaml")
    provenance = {
        "run_id": run_id,
        "seed": seed,
        "config_hash": config_hash(cfg),
        "tier": cfg.get("tier"),
        "dataset": cfg.get("dataset"),
        "condition": cfg.get("condition", "multiple"),
        "status": "scaffold-created",
        "note": "Experiment-specific data/model adapters must be configured before full research execution.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    pd.DataFrame([provenance]).to_csv(out / "metrics.csv", index=False)
    (out / "predictions.jsonl").write_text("", encoding="utf-8")
    return out
