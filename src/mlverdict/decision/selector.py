"""Hard constraints first, then soft scores. Never a hard-coded favorite model."""

from __future__ import annotations

from mlverdict.core.enums import DecisionStatus
from mlverdict.core.types import EvaluationResult, ModelDecision, MetricPlan, Rejection
from mlverdict.decision.tradeoffs import describe_tradeoffs


def select_model(
    evaluations: tuple[EvaluationResult, ...],
    metric_plan: MetricPlan,
    *,
    assumptions: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
) -> ModelDecision:
    if not evaluations:
        return ModelDecision(
            status=DecisionStatus.BLOCKED,
            selected_model=None,
            reasons=("No successful experiments produced evaluation evidence.",),
            tradeoffs=(),
            rejected=(),
            assumptions=assumptions,
            limitations=limitations + ("No model could be evaluated.",),
            primary_metric=metric_plan.primary.name,
        )

    rejected: list[Rejection] = []
    eligible = []
    for ev in evaluations:
        if ev.passes_constraints:
            eligible.append(ev)
        else:
            rejected.append(Rejection(ev.model_name, "; ".join(ev.constraint_failures)))

    if not eligible:
        return ModelDecision(
            status=DecisionStatus.BLOCKED,
            selected_model=None,
            reasons=("Every evaluated model failed a hard production constraint.",),
            tradeoffs=(),
            rejected=tuple(rejected),
            assumptions=assumptions,
            limitations=limitations + ("Relax constraints or collect a cheaper model family.",),
            primary_metric=metric_plan.primary.name,
        )

    ranked = sorted(
        eligible,
        key=lambda e: (e.composite_score, e.stability_score, e.primary_score),
        reverse=True,
    )
    winner = ranked[0]
    for ev in ranked[1:]:
        rejected.append(
            Rejection(
                ev.model_name,
                (
                    f"lower composite evidence (score={ev.composite_score:.4f} vs {winner.composite_score:.4f}); "
                    f"primary={ev.primary_score:.4f}, stability={ev.stability_score:.3f}, "
                    f"latency={ev.latency_ms:.2f}ms"
                ),
            )
        )

    reasons = [
        f"Selected {winner.model_name} after hard-constraint filtering and multi-criteria ranking.",
        f"Primary metric '{metric_plan.primary.name}' validation score={winner.primary_score:.4f}.",
        f"Stability={winner.stability_score:.3f}, generalization={winner.generalization_score:.3f}, "
        f"latency={winner.latency_ms:.2f}ms, complexity={winner.complexity}.",
        f"Composite evidence score={winner.composite_score:.4f} (ranking aid, not a published accuracy).",
    ]
    if metric_plan.user_override:
        reasons.append("User-selected primary metric was respected.")

    return ModelDecision(
        status=DecisionStatus.DECIDED,
        selected_model=winner.model_name,
        reasons=tuple(reasons),
        tradeoffs=describe_tradeoffs(winner, evaluations),
        rejected=tuple(rejected),
        assumptions=assumptions,
        limitations=limitations,
        primary_metric=metric_plan.primary.name,
        selected_primary_score=winner.primary_score,
    )
