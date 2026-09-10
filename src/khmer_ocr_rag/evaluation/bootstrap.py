from __future__ import annotations

from collections.abc import Callable

import numpy as np


def paired_bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 5000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape")
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    if n == 0:
        raise ValueError("No observations")
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats[i] = statistic(diffs[idx])
    alpha = (1 - confidence) / 2
    return {
        "difference": float(statistic(diffs)),
        "ci_low": float(np.quantile(stats, alpha)),
        "ci_high": float(np.quantile(stats, 1 - alpha)),
    }
