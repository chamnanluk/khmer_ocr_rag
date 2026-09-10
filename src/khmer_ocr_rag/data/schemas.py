from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Tier = Literal["A", "B"]
Split = Literal["train", "val", "test"]
Modality = Literal["document", "scene", "handwritten"]


@dataclass(slots=True)
class TextLineRecord:
    sample_id: str
    tier: Tier
    dataset: str
    split: Split
    modality: Modality
    image_path: str
    text_gold: str
    text_gold_nfc: str | None = None
    text_gold_segmented: str | None = None
    boundary_source: str = "missing"
    document_id: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PassageRecord:
    passage_id: str
    document_id: str
    text_gold: str
    text_condition: str
    condition: str
    source_span: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QARecord:
    query_id: str
    question_km: str
    answer_gold: str
    relevant_passage_ids: list[str]
    supporting_passage_ids: list[str] = field(default_factory=list)
    answer_type: str = "free_text"
    annotator_status: str = "unverified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
