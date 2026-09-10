from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from .unicode import ZERO_WIDTH_SPACE

KHMER_START = 0x1780
KHMER_END = 0x17FF
COENG = "\u17d2"


def is_khmer_char(ch: str) -> bool:
    return len(ch) == 1 and KHMER_START <= ord(ch) <= KHMER_END


def is_combining(ch: str) -> bool:
    return unicodedata.category(ch).startswith("M")


def khmer_clusters(text: str, keep_boundaries: bool = True) -> list[str]:
    """Practical Khmer orthographic cluster tokenizer.

    This keeps COENG+following consonant and combining marks with the current cluster.
    It is intended as a reproducible research baseline, not as a claim of exact equivalence
    to every published KCC rule set. Official tokenizer code should replace this function
    through the same API when available.
    """
    clusters: list[str] = []
    cur = ""
    expect_after_coeng = False

    def flush() -> None:
        nonlocal cur
        if cur:
            clusters.append(cur)
            cur = ""

    for ch in text:
        if ch == ZERO_WIDTH_SPACE:
            flush()
            if keep_boundaries:
                clusters.append(ch)
            expect_after_coeng = False
            continue

        if not is_khmer_char(ch):
            flush()
            clusters.append(ch)
            expect_after_coeng = False
            continue

        if not cur:
            cur = ch
            expect_after_coeng = ch == COENG
            continue

        if expect_after_coeng:
            cur += ch
            expect_after_coeng = False
            continue

        if ch == COENG:
            cur += ch
            expect_after_coeng = True
            continue

        if is_combining(ch):
            cur += ch
            continue

        # Khmer signs/vowels often have category Lo rather than M. Attach characters that
        # Unicode identifies as dependent vowel/sign via their name; start a new cluster
        # for independent letters/numerals.
        name = unicodedata.name(ch, "")
        if "VOWEL SIGN" in name or "SIGN" in name:
            cur += ch
        else:
            flush()
            cur = ch

    flush()
    return clusters


def join_clusters(items: Iterable[str]) -> str:
    return "".join(items)
