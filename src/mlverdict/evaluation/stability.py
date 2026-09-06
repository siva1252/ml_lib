"""Stability from fold scores: low variance and a bounded worst fold."""

from __future__ import annotations

import math


def stability_score(std: float, mean: float, min_score: float) -> float:
    if not math.isfinite(std) or not math.isfinite(mean):
        return 0.0
    relative = abs(std) / (abs(mean) + 1e-9)
    spread = abs(mean - min_score) if math.isfinite(min_score) else relative
    raw = 1.0 / (1.0 + relative + 0.5 * abs(spread))
    return float(max(0.0, min(1.0, raw)))
