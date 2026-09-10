from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from khmer_ocr_rag.data.kcc import COENG, is_khmer_char, khmer_clusters
from khmer_ocr_rag.data.unicode import ZERO_WIDTH_SPACE


class ErrorFamily(str, Enum):
    BASE = "base"
    SUBSCRIPT = "subscript"
    VOWEL = "vowel"
    DIACRITIC = "diacritic"
    KCC = "kcc"
    UNICODE = "unicode"
    BOUNDARY_INSERT = "boundary_insert"
    BOUNDARY_DELETE = "boundary_delete"


@dataclass(frozen=True)
class Edit:
    error_family: str
    start_char: int
    end_char: int
    before: str
    after: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CorruptionResult:
    text: str
    edits: list[Edit]
    target_rate: float
    realized_rate: float


def _base_positions(text: str) -> list[int]:
    # Modern Khmer consonant code points occupy U+1780..U+17A2.
    return [i for i, ch in enumerate(text) if 0x1780 <= ord(ch) <= 0x17A2]


def _subscript_spans(text: str) -> list[tuple[int, int]]:
    return [(i, i + 2) for i, ch in enumerate(text[:-1]) if ch == COENG and is_khmer_char(text[i + 1])]


def _vowel_positions(text: str) -> list[int]:
    return [i for i, ch in enumerate(text) if "VOWEL SIGN" in unicodedata.name(ch, "")]


def _diacritic_positions(text: str) -> list[int]:
    out = []
    for i, ch in enumerate(text):
        name = unicodedata.name(ch, "")
        if ch != COENG and "SIGN" in name and "VOWEL SIGN" not in name:
            out.append(i)
    return out


def _cluster_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for cluster in khmer_clusters(text, keep_boundaries=True):
        start = offset
        offset += len(cluster)
        if cluster != ZERO_WIDTH_SPACE and any(is_khmer_char(c) for c in cluster):
            spans.append((start, offset))
    return spans


def _unicode_reorder_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for start, end in _cluster_spans(text):
        cluster = text[start:end]
        # Require at least two non-base components to create a controlled ordering error.
        if len(cluster) >= 3:
            spans.append((start, end))
    return spans


def _visible_length(text: str) -> int:
    return max(1, len(text.replace(ZERO_WIDTH_SPACE, "")))


def _target_edits(text: str, rate: float, candidate_count: int) -> int:
    return min(candidate_count, round(rate * _visible_length(text)))


def _choose_indices(rng: np.random.Generator, n_candidates: int, n: int) -> list[int]:
    if n <= 0 or n_candidates == 0:
        return []
    return sorted(rng.choice(n_candidates, size=n, replace=False).tolist())


def corrupt_text(
    text: str,
    family: ErrorFamily | str,
    rate: float,
    rng: np.random.Generator,
    replacement_pool: list[str] | None = None,
) -> CorruptionResult:
    """Inject one Khmer-specific error family and return an auditable edit log.

    The default replacement behavior is deletion, which is deterministic and avoids
    inventing linguistically implausible substitutions. For publication experiments,
    pass confusion-matrix-derived `replacement_pool` values or extend this function with
    Tier-B conditional replacement distributions.
    """
    family = ErrorFamily(family)
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")

    edits: list[Edit] = []

    if family == ErrorFamily.BOUNDARY_DELETE:
        candidates = [i for i, ch in enumerate(text) if ch == ZERO_WIDTH_SPACE]
        chosen = _choose_indices(rng, len(candidates), _target_edits(text, rate, len(candidates)))
        chars = list(text)
        for ci in reversed(chosen):
            pos = candidates[ci]
            edits.append(Edit(family.value, pos, pos + 1, ZERO_WIDTH_SPACE, ""))
            del chars[pos]
        out = "".join(chars)

    elif family == ErrorFamily.BOUNDARY_INSERT:
        candidates = [
            i for i in range(1, len(text))
            if text[i - 1] != ZERO_WIDTH_SPACE and text[i] != ZERO_WIDTH_SPACE
        ]
        chosen = _choose_indices(rng, len(candidates), _target_edits(text, rate, len(candidates)))
        chars = list(text)
        for ci in reversed(chosen):
            pos = candidates[ci]
            edits.append(Edit(family.value, pos, pos, "", ZERO_WIDTH_SPACE))
            chars.insert(pos, ZERO_WIDTH_SPACE)
        out = "".join(chars)

    elif family == ErrorFamily.SUBSCRIPT:
        spans = _subscript_spans(text)
        chosen = _choose_indices(rng, len(spans), _target_edits(text, rate, len(spans)))
        out = text
        for ci in reversed(chosen):
            start, end = spans[ci]
            before = text[start:end]
            # Remove the subscript unit COENG+consonant as a structural corruption.
            out = out[:start] + out[end:]
            edits.append(Edit(family.value, start, end, before, ""))

    elif family == ErrorFamily.KCC:
        spans = _cluster_spans(text)
        chosen = _choose_indices(rng, len(spans), _target_edits(text, rate, len(spans)))
        out = text
        for ci in reversed(chosen):
            start, end = spans[ci]
            before = text[start:end]
            choices = [x for x in (replacement_pool or []) if x != before]
            after = str(rng.choice(choices)) if choices else ""
            out = out[:start] + after + out[end:]
            edits.append(Edit(family.value, start, end, before, after))

    elif family == ErrorFamily.UNICODE:
        spans = _unicode_reorder_spans(text)
        chosen = _choose_indices(rng, len(spans), _target_edits(text, rate, len(spans)))
        out = text
        for ci in reversed(chosen):
            start, end = spans[ci]
            cluster = list(text[start:end])
            before = "".join(cluster)
            # Keep the leading base stable and swap two trailing code points.
            cluster[-1], cluster[-2] = cluster[-2], cluster[-1]
            after = "".join(cluster)
            out = out[:start] + after + out[end:]
            edits.append(Edit(family.value, start, end, before, after))

    else:
        if family == ErrorFamily.BASE:
            candidates = _base_positions(text)
        elif family == ErrorFamily.VOWEL:
            candidates = _vowel_positions(text)
        elif family == ErrorFamily.DIACRITIC:
            candidates = _diacritic_positions(text)
        else:
            raise AssertionError(family)

        chosen = _choose_indices(rng, len(candidates), _target_edits(text, rate, len(candidates)))
        chars = list(text)
        for ci in reversed(chosen):
            pos = candidates[ci]
            before = chars[pos]
            choices = [x for x in (replacement_pool or []) if x != before]
            after = str(rng.choice(choices)) if choices else ""
            chars[pos] = after
            edits.append(Edit(family.value, pos, pos + 1, before, after))
        out = "".join(chars)

    edits = list(reversed(edits))
    return CorruptionResult(
        text=out,
        edits=edits,
        target_rate=rate,
        realized_rate=len(edits) / _visible_length(text),
    )
