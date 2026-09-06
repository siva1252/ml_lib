"""Choose primary and secondary metrics from problem + DNA + user override."""

from __future__ import annotations

from mlverdict.core.enums import ProblemType
from mlverdict.core.exceptions import ConfigurationError
from mlverdict.core.types import DatasetDNA, MetricPlan, ProblemDefinition
from mlverdict.metrics.registry import CLASSIFICATION, REGRESSION, get_metric


def select_metric_plan(
    dna: DatasetDNA,
    problem: ProblemDefinition,
    *,
    user_metric: str | None = None,
    false_positive_cost: float | None = None,
    false_negative_cost: float | None = None,
) -> MetricPlan:
    if problem.problem_type is None:
        raise ConfigurationError("Cannot select metrics without a decided problem type.")

    user_override = user_metric is not None
    evidence: list[str] = [
        f"problem={problem.problem_type.value}",
        f"imbalance_ratio={dna.imbalance_ratio}",
        f"target_kind={dna.target_kind}",
    ]

    if problem.is_regression:
        primary = get_metric(user_metric) if user_metric else REGRESSION["rmse"]
        if user_metric and primary.name not in REGRESSION:
            raise ConfigurationError(f"Metric '{user_metric}' is not a regression metric.")
        secondary = (REGRESSION["mae"], REGRESSION["r2"], REGRESSION["mape"])
        evidence.append("regression default primary=rmse" if not user_metric else f"user metric={user_metric}")
        return MetricPlan(primary=primary, secondary=secondary, evidence=tuple(evidence), user_override=user_override)

    imbalanced = bool(dna.imbalance_ratio and dna.imbalance_ratio >= 3.0)
    if user_metric:
        primary = get_metric(user_metric)
        if primary.name not in CLASSIFICATION:
            raise ConfigurationError(f"Metric '{user_metric}' is not a classification metric.")
        evidence.append(f"user metric={user_metric}")
    elif false_negative_cost and false_positive_cost and false_negative_cost > 2 * false_positive_cost:
        primary = CLASSIFICATION["recall"]
        evidence.append("FN cost dominates → recall")
    elif false_positive_cost and false_negative_cost and false_positive_cost > 2 * false_negative_cost:
        primary = CLASSIFICATION["precision"]
        evidence.append("FP cost dominates → precision")
    elif imbalanced and problem.problem_type == ProblemType.BINARY_CLASSIFICATION:
        primary = CLASSIFICATION["pr_auc"]
        evidence.append("imbalanced binary → pr_auc (accuracy would be misleading)")
    elif problem.problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
        primary = CLASSIFICATION["f1"]
        evidence.append("multiclass default → f1 (macro via sklearn f1)")
    else:
        primary = CLASSIFICATION["roc_auc"]
        evidence.append("balanced binary → roc_auc")

    secondary = (
        CLASSIFICATION["accuracy"],
        CLASSIFICATION["f1"],
        CLASSIFICATION["precision"],
        CLASSIFICATION["recall"],
        CLASSIFICATION["roc_auc"] if problem.problem_type == ProblemType.BINARY_CLASSIFICATION else CLASSIFICATION["balanced_accuracy"],
        CLASSIFICATION["log_loss"],
    )
    if problem.problem_type == ProblemType.BINARY_CLASSIFICATION:
        secondary = secondary + (CLASSIFICATION["pr_auc"],)
    return MetricPlan(primary=primary, secondary=secondary, evidence=tuple(evidence), user_override=user_override)
