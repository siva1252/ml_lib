"""Normalize primary scores onto a 0-1 band for soft ranking only."""

from __future__ import annotations

import math


def minmax(values: list[float]) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0 for _ in values]
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-12:
        return [1.0 if math.isfinite(v) else 0.0 for v in values]
    return [0.0 if not math.isfinite(v) else (v - lo) / (hi - lo) for v in values]


def noise_aware_scale(values: list[float], stds: list[float], n_folds: list[int]) -> list[float]:
    """Map primary scores onto 0-1 without inflating differences smaller than CV noise.

    Plain min-max turns a 0.004 gap into a 0-vs-1 swing. Here the denominator is at least
    four standard errors of the fold scores, so models that are statistically tied score
    close to each other and the other criteria (stability, latency, simplicity) decide.
    """
    finite = [(v, s, k) for v, s, k in zip(values, stds, n_folds) if math.isfinite(v)]
    if not finite:
        return [0.0 for _ in values]
    best = max(v for v, _, _ in finite)
    lo = min(v for v, _, _ in finite)
    ses = sorted((s if math.isfinite(s) else 0.0) / math.sqrt(max(k, 1)) for _, s, k in finite)
    median_se = ses[len(ses) // 2]
    denom = max(best - lo, 4.0 * median_se, 1e-12)
    return [
        0.0 if not math.isfinite(v) else max(0.0, min(1.0, 1.0 - (best - v) / denom)) for v in values
    ]
