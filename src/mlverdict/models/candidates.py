"""Intelligent candidate selection. Do not blindly train every catalog entry."""

from __future__ import annotations

from mlverdict.core.config import Constraints
from mlverdict.core.enums import DatasetScale, ModelFamily
from mlverdict.core.types import CandidateModel, CandidateSet, DatasetDNA, ExclusionRecord, MetricPlan, ProblemDefinition
from mlverdict.models.compatibility import compatible
from mlverdict.models.registry import CATALOG


def select_candidates(
    dna: DatasetDNA,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
    constraints: Constraints | None = None,
) -> CandidateSet:
    included: list[CandidateModel] = []
    excluded: list[ExclusionRecord] = []

    prefer_linear = dna.scale in {DatasetScale.TINY, DatasetScale.SMALL} and dna.numeric_fraction >= 0.6
    prefer_trees = bool(dna.high_cardinality_columns) or dna.categorical_fraction >= 0.4
    tight_latency = bool(constraints and constraints.max_latency_ms is not None and constraints.max_latency_ms < 40)

    for entry in CATALOG:
        ok, reason = compatible(entry, dna, problem, constraints)
        if not ok:
            excluded.append(ExclusionRecord(entry["name"], reason))
            continue

        why = [f"compatible with {problem.problem_type.value}"]
        if entry["family"] == ModelFamily.LINEAR and prefer_linear:
            why.append("numeric-heavy / small data favors a linear baseline")
        if entry["family"] in {ModelFamily.TREE, ModelFamily.BOOSTING} and prefer_trees:
            why.append("categorical / high-cardinality structure favors trees")
        if tight_latency and entry["complexity"] == "low":
            why.append("tight latency budget favors a cheap model")
        why.append(f"primary metric={metric_plan.primary.name}")

        # Skip extra trees on tiny data when RF is already in — keep a shortlist.
        if entry["key"] == "extra_trees" and dna.scale == DatasetScale.TINY:
            excluded.append(ExclusionRecord(entry["name"], "redundant with Random Forest on tiny data"))
            continue
        if entry["key"] == "hist_gradient_boosting" and dna.scale == DatasetScale.TINY:
            excluded.append(ExclusionRecord(entry["name"], "skipped on tiny data"))
            continue
        if entry["key"] in {"xgboost", "lightgbm"} and dna.scale == DatasetScale.SMALL and prefer_trees:
            # Keep CatBoost or HGB, not every booster.
            if any(c.estimator_key == "hist_gradient_boosting" for c in included):
                excluded.append(ExclusionRecord(entry["name"], "HGB already covers boosting on small data"))
                continue

        included.append(
            CandidateModel(
                name=entry["name"],
                family=entry["family"],
                estimator_key=entry["key"],
                why_included="; ".join(why),
                params={},
                available=True,
                explainable=entry["explainable"],
                complexity=entry["complexity"],
            )
        )

    if not included:
        # Last resort: always try the linear adapter if the problem is decided.
        fallback_key = "ridge" if problem.is_regression else "logistic_regression"
        fallback_name = "Ridge" if problem.is_regression else "Logistic Regression"
        included.append(
            CandidateModel(
                name=fallback_name,
                family=ModelFamily.LINEAR,
                estimator_key=fallback_key,
                why_included="fallback linear model; all other candidates were filtered",
                explainable=True,
                complexity="low",
            )
        )

    return CandidateSet(included=tuple(included), excluded=tuple(excluded))
