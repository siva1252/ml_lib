"""Human-readable tradeoff lines between the winner and rejected alternatives."""

from __future__ import annotations

from mlverdict.core.types import EvaluationResult


def describe_tradeoffs(winner: EvaluationResult, others: tuple[EvaluationResult, ...]) -> tuple[str, ...]:
    lines: list[str] = []
    for alt in others:
        if alt.model_name == winner.model_name:
            continue
        if alt.primary_score > winner.primary_score + 1e-9:
            lines.append(
                f"{alt.model_name} scored higher on the primary metric "
                f"({alt.primary_score:.4f} vs {winner.primary_score:.4f}) but lost on constraints or stability."
            )
        elif winner.stability_score > alt.stability_score + 0.05:
            lines.append(
                f"{winner.model_name} is more stable ({winner.stability_score:.3f} vs {alt.stability_score:.3f}) "
                f"than {alt.model_name}."
            )
        elif winner.latency_ms + 1e-6 < alt.latency_ms:
            lines.append(
                f"{winner.model_name} is faster ({winner.latency_ms:.2f}ms vs {alt.latency_ms:.2f}ms) "
                f"than {alt.model_name}."
            )
    return tuple(lines)
