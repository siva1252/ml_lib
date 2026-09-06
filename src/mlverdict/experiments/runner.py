"""Experiment orchestrator: baselines, then bounded HPO on the shortlist."""

from __future__ import annotations

import pandas as pd

from mlverdict.core.config import VerdictConfig
from mlverdict.core.types import (
    CandidateModel,
    DatasetDNA,
    DatasetProfile,
    ExperimentResult,
    MetricPlan,
    ProblemDefinition,
    ValidationPlan,
)
from mlverdict.experiments.baseline import run_baselines
from mlverdict.experiments.hpo import optimize_candidate, select_promising


def run_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: tuple[CandidateModel, ...],
    profile: DatasetProfile,
    dna: DatasetDNA,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
    validation: ValidationPlan,
    config: VerdictConfig,
    *,
    groups: pd.Series | None = None,
    extra_drop: tuple[str, ...] = (),
    enable_hpo: bool = True,
) -> tuple[ExperimentResult, ...]:
    baselines = run_baselines(
        X,
        y,
        candidates,
        profile,
        dna,
        problem,
        metric_plan,
        validation,
        config,
        groups=groups,
        extra_drop=extra_drop,
    )
    if not enable_hpo or config.max_hpo_trials <= 0:
        return baselines

    by_name = {c.name: c for c in candidates}
    promising = select_promising(baselines, config.n_hpo_candidates)
    optimized: list[ExperimentResult] = []
    optimized_names: set[str] = set()
    for base in promising:
        candidate = by_name.get(base.model_name)
        if candidate is None:
            continue
        result = optimize_candidate(
            candidate,
            X,
            y,
            profile,
            dna,
            problem,
            metric_plan,
            validation,
            config,
            groups=groups,
            extra_drop=extra_drop,
        )
        optimized.append(result)
        optimized_names.add(result.model_name)

    merged = [r for r in baselines if r.model_name not in optimized_names]
    merged.extend(optimized)
    return tuple(merged)
