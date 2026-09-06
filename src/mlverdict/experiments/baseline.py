"""Baseline pass: one experiment per included candidate, no HPO."""

from __future__ import annotations

import pandas as pd

from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import ModelFamily
from mlverdict.core.types import (
    CandidateModel,
    DatasetDNA,
    DatasetProfile,
    ExperimentResult,
    MetricPlan,
    ProblemDefinition,
    ValidationPlan,
)
from mlverdict.experiments.cv import run_cross_validation
from mlverdict.preprocessing.pipeline import build_pipeline
from mlverdict.preprocessing.planner import plan_preprocessing


def run_baselines(
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
) -> tuple[ExperimentResult, ...]:
    results: list[ExperimentResult] = []
    assert problem.problem_type is not None
    for candidate in candidates:
        plan = plan_preprocessing(
            profile,
            dna,
            candidate.family if isinstance(candidate.family, ModelFamily) else ModelFamily(candidate.family),
            config,
            extra_drop=extra_drop,
        )
        pipeline = build_pipeline(plan, candidate, problem.problem_type, config.random_state)
        result = run_cross_validation(
            pipeline,
            X,
            y,
            candidate,
            problem,
            metric_plan,
            validation,
            random_state=config.random_state,
            groups=groups,
            latency_probe_rows=config.latency_probe_rows,
        )
        results.append(result)
    return tuple(results)
