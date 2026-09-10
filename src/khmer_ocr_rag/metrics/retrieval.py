from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def hit_rate_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    return float(bool(set(ranked_ids[:k]) & relevant))


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(ranked_ids[:k]) & relevant) / k


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    limit = ranked_ids if k is None else ranked_ids[:k]
    for i, item in enumerate(limit, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked_ids[:k], 1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def aggregate_query_metrics(rows: Iterable[dict]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {key: sum(float(r[key]) for r in rows) / len(rows) for key in keys}
