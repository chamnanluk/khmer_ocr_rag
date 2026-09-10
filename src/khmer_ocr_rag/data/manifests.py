from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

from .schemas import QARecord, TextLineRecord
from .unicode import normalize_khmer


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return records


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_textline_manifest(path: str | Path) -> list[TextLineRecord]:
    p = Path(path)
    rows: list[dict]
    if p.suffix.lower() == ".jsonl":
        rows = read_jsonl(p)
    elif p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError("Manifest must be .jsonl or .csv")

    output: list[TextLineRecord] = []
    seen: set[str] = set()
    for row in rows:
        sid = str(row["sample_id"])
        if sid in seen:
            raise ValueError(f"Duplicate sample_id: {sid}")
        seen.add(sid)
        text = str(row["text_gold"])
        row["text_gold_nfc"] = row.get("text_gold_nfc") or normalize_khmer(text)
        output.append(TextLineRecord(**row))
    return output


def load_qa(path: str | Path) -> list[QARecord]:
    return [QARecord(**r) for r in read_jsonl(path)]
