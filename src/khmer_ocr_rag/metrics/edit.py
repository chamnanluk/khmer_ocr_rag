from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def levenshtein(a: Sequence[T], b: Sequence[T]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[T], hypothesis: Sequence[T]) -> float:
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return levenshtein(reference, hypothesis) / len(reference)
