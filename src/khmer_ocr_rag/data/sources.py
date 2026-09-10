from __future__ import annotations

import csv
import json
from pathlib import Path

from .manifests import read_jsonl


def read_tier_a_sources(path: str | Path) -> list[dict]:
    """Read local DATA-01 source records without performing network access."""
    source_path = Path(path)
    if source_path.suffix.lower() == ".jsonl":
        return read_jsonl(source_path)
    if source_path.suffix.lower() == ".csv":
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle))
        for line_number, record in enumerate(records, start=2):
            metadata = record.get("metadata")
            if metadata:
                try:
                    record["metadata"] = json.loads(metadata)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid metadata JSON at {source_path}:{line_number}"
                    ) from exc
            else:
                record["metadata"] = {}
            if not record.get("text_gold_segmented"):
                record["text_gold_segmented"] = None
        return records
    raise ValueError("Tier A source input must be .jsonl or .csv")
