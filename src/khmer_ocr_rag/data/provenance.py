from __future__ import annotations

import hashlib
from pathlib import Path

from khmer_ocr_rag.utils.repro import config_hash


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of an immutable input or derived artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_provenance(
    *,
    input_path: str | Path,
    output_path: str | Path,
    config: dict,
    counts_by_split: dict[str, int],
) -> dict:
    """Build deterministic provenance for a DATA-01 manifest derivation."""
    return {
        "artifact_type": "tier_a_manifest",
        "dataset": config["dataset"],
        "seed": int(config["seed"]),
        "config_hash": config_hash(config),
        "input_path": Path(input_path).as_posix(),
        "input_sha256": sha256_file(input_path),
        "output_path": Path(output_path).as_posix(),
        "output_sha256": sha256_file(output_path),
        "counts_by_split": dict(counts_by_split),
    }
