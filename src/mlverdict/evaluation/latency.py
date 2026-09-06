"""Latency helpers."""

from __future__ import annotations


def latency_score(latency_ms: float, budget_ms: float | None) -> float:
    if latency_ms != latency_ms or latency_ms == float("inf"):
        return 0.0
    ref = budget_ms if budget_ms else 200.0
    return float(max(0.0, min(1.0, 1.0 - (latency_ms / (ref * 3.0)))))
