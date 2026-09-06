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
