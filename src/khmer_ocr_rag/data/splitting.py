from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict

from .unicode import normalize_khmer

SPLITS = ("train", "val", "test")
BOUNDARY_SOURCES = {"human", "licensed_manual_corpus", "auto_silver", "missing"}
MANUAL_BOUNDARY_SOURCES = {"human", "licensed_manual_corpus"}
MODALITIES = {"document", "scene", "handwritten"}


def _validate_ratios(split_ratios: dict[str, float]) -> dict[str, float]:
    if set(split_ratios) != set(SPLITS):
        raise ValueError(f"split_ratios must contain exactly: {', '.join(SPLITS)}")
    ratios = {split: float(split_ratios[split]) for split in SPLITS}
    if any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios cannot be negative")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1.0")
    return ratios


def _component_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    positive_splits = [split for split in SPLITS if ratios[split] > 0]
    reserve_each = total >= len(positive_splits)
    counts = {split: int(reserve_each and split in positive_splits) for split in SPLITS}
    remaining = total - sum(counts.values())
    raw = {split: remaining * ratios[split] for split in SPLITS}
    additions = {split: math.floor(raw[split]) for split in SPLITS}
    for split in SPLITS:
        counts[split] += additions[split]
    remaining = total - sum(counts.values())
    priority = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - additions[split]), SPLITS.index(split)),
    )
    for split in priority[:remaining]:
        counts[split] += 1
    return counts


def _stable_key(seed: int, document_ids: set[str]) -> str:
    grouped_ids = "\0".join(sorted(document_ids))
    value = f"{seed}\0{grouped_ids}".encode()
    return hashlib.sha256(value).hexdigest()


def _normalize_record(
    record: dict,
    *,
    dataset: str,
    default_license: str | None,
) -> dict:
    sample_value = record.get("sample_id")
    document_value = record.get("document_id")
    text_value = record.get("text_gold", record.get("text"))
    sample_id = "" if sample_value is None else str(sample_value).strip()
    document_id = "" if document_value is None else str(document_value).strip()
    text_gold = "" if text_value is None else str(text_value)
    license_name = str(record.get("license") or default_license or "").strip()
    modality = str(record.get("modality") or "document")
    boundary_source = str(record.get("boundary_source") or "missing")
    metadata = record.get("metadata") or {}

    if not sample_id:
        raise ValueError("Every source record requires a non-empty sample_id")
    if not document_id:
        raise ValueError(f"Record {sample_id} requires a non-empty document_id")
    if not text_gold.strip():
        raise ValueError(f"Record {sample_id} requires non-empty text_gold")
    if not license_name:
        raise ValueError(f"Record {sample_id} requires license provenance")
    if record.get("tier", "A") != "A":
        raise ValueError(f"Record {sample_id} must belong to Tier A")
    if record.get("dataset") and record["dataset"] != dataset:
        raise ValueError(f"Record {sample_id} dataset does not match {dataset!r}")
    if modality not in MODALITIES:
        raise ValueError(f"Record {sample_id} has invalid modality: {modality}")
    if boundary_source not in BOUNDARY_SOURCES:
        raise ValueError(f"Record {sample_id} has invalid boundary_source: {boundary_source}")
    if not isinstance(metadata, dict):
        raise TypeError(f"Record {sample_id} metadata must be an object")

    segmented = record.get("text_gold_segmented")
    boundary_generation = str(metadata.get("boundary_generation", "")).lower()
    if (
        boundary_generation in {"auto", "automatic", "model", "silver"}
        and boundary_source in MANUAL_BOUNDARY_SOURCES
    ):
        raise ValueError(f"Automatic boundaries cannot be labeled {boundary_source}: {sample_id}")
    if segmented is not None and boundary_source == "missing":
        raise ValueError(f"Record {sample_id} has segmented text without boundary provenance")

    return {
        "sample_id": sample_id,
        "tier": "A",
        "dataset": dataset,
        "split": "train",  # Assigned after source components are built.
        "modality": modality,
        "image_path": str(record.get("image_path") or ""),
        "text_gold": text_gold,
        "text_gold_nfc": normalize_khmer(text_gold),
        "text_gold_segmented": normalize_khmer(str(segmented)) if segmented is not None else None,
        "boundary_source": boundary_source,
        "document_id": document_id,
        "license": license_name,
        "metadata": dict(metadata),
    }


def _source_components(records: list[dict]) -> list[set[str]]:
    """Join source documents that share normalized text to prevent split leakage."""
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    documents_by_text: dict[str, set[str]] = defaultdict(set)
    for record in records:
        document_id = record["document_id"]
        find(document_id)
        documents_by_text[record["text_gold_nfc"]].add(document_id)

    for document_ids in documents_by_text.values():
        ordered = sorted(document_ids)
        for document_id in ordered[1:]:
            union(ordered[0], document_id)

    components: dict[str, set[str]] = defaultdict(set)
    for document_id in parent:
        components[find(document_id)].add(document_id)
    return list(components.values())


def prepare_tier_a_records(
    records: list[dict],
    *,
    dataset: str,
    default_license: str | None,
    seed: int,
    split_ratios: dict[str, float] | None = None,
) -> tuple[list[dict], dict]:
    """Validate and deterministically assign Tier A records to source-level splits.

    Documents sharing NFC-normalized text are treated as one connected component. This can
    make achieved row ratios differ slightly from requested ratios, but guarantees that a
    duplicate sentence cannot leak across splits.
    """
    dataset = dataset.strip()
    if not dataset:
        raise ValueError("dataset must be non-empty")
    ratios = _validate_ratios(split_ratios or {"train": 0.8, "val": 0.1, "test": 0.1})
    if not records:
        raise ValueError("Tier A source input must contain at least one record")
    normalized = [
        _normalize_record(record, dataset=dataset, default_license=default_license)
        for record in records
    ]

    sample_ids = [record["sample_id"] for record in normalized]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate sample_id: {duplicate_ids[0]}")

    components = sorted(_source_components(normalized), key=lambda ids: _stable_key(seed, ids))
    requested_counts = _component_counts(len(components), ratios)
    split_by_document: dict[str, str] = {}
    offset = 0
    for split in SPLITS:
        stop = offset + requested_counts[split]
        for component in components[offset:stop]:
            for document_id in component:
                split_by_document[document_id] = split
        offset = stop

    for record in normalized:
        record["split"] = split_by_document[record["document_id"]]
    normalized.sort(key=lambda record: record["sample_id"])

    text_splits: dict[str, set[str]] = defaultdict(set)
    for record in normalized:
        text_splits[record["text_gold_nfc"]].add(record["split"])
    leaks = sum(len(splits) > 1 for splits in text_splits.values())
    if leaks:
        raise AssertionError(f"Internal error: detected {leaks} cross-split normalized-text leaks")

    counts_by_split = Counter(record["split"] for record in normalized)
    report = {
        "dataset": dataset,
        "seed": int(seed),
        "requested_split_ratios": ratios,
        "counts_by_split": {split: counts_by_split.get(split, 0) for split in SPLITS},
        "source_components_by_split": requested_counts,
        "source_component_count": len(components),
        "record_count": len(normalized),
        "cross_split_text_leaks": leaks,
        "assumptions": [
            "document_id identifies the indivisible source-level split group",
            "documents sharing NFC-normalized text are joined before splitting",
            "component allocation targets source-component ratios, not exact row ratios",
        ],
    }
    return normalized, report
