from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean, median


@dataclass(frozen=True)
class TimingSummary:
    mean_ms: float
    median_ms: float
    p95_ms: float
    throughput_per_s: float
    trials: int


def benchmark_callable(fn, warmup: int = 10, trials: int = 100, synchronize=None) -> TimingSummary:
    for _ in range(warmup):
        fn()
    if synchronize:
        synchronize()
    values = []
    for _ in range(trials):
        if synchronize:
            synchronize()
        start = time.perf_counter()
        fn()
        if synchronize:
            synchronize()
        values.append((time.perf_counter() - start) * 1000)
    values_sorted = sorted(values)
    p95 = values_sorted[min(len(values_sorted) - 1, int(0.95 * len(values_sorted)))]
    avg = mean(values)
    return TimingSummary(avg, median(values), p95, 1000.0 / avg if avg else float("inf"), trials)
