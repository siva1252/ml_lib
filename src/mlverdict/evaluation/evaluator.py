"""Score each successful experiment across predictive, stability, gap, latency, complexity."""

from __future__ import annotations

import math

from mlverdict.core.config import Constraints, VerdictConfig
from mlverdict.core.enums import ExperimentStatus
from mlverdict.core.types import EvaluationResult, ExperimentResult
from mlverdict.evaluation.comparison import minmax
from mlverdict.evaluation.latency import latency_score
from mlverdict.evaluation.stability import stability_score

_COMPLEXITY_SCORE = {"low": 1.0, "medium": 0.6, "high": 0.3}


def _constraint_failures(exp: ExperimentResult, constraints: Constraints | None) -> tuple[str, ...]:
    if constraints is None:
        return ()
    fails: list[str] = []
    if constraints.max_latency_ms is not None and exp.infer_latency_ms > constraints.max_latency_ms:
        fails.append(
            f"latency {exp.infer_latency_ms:.2f}ms exceeds max_latency_ms={constraints.max_latency_ms}"
        )
    if constraints.max_train_time_seconds is not None and exp.train_time_seconds > constraints.max_train_time_seconds:
        fails.append(
            f"train_time {exp.train_time_seconds:.2f}s exceeds max_train_time_seconds={constraints.max_train_time_seconds}"
        )
    if constraints.require_explainable and not exp.explainable:
        fails.append("model is not in the explainable set")
    if constraints.max_complexity == "low" and exp.complexity != "low":
        fails.append(f"complexity {exp.complexity} exceeds max_complexity=low")
    if constraints.max_complexity == "medium" and exp.complexity == "high":
        fails.append("complexity high exceeds max_complexity=medium")
    return tuple(fails)


def evaluate_experiments(
    experiments: tuple[ExperimentResult, ...],
    config: VerdictConfig,
    constraints: Constraints | None,
) -> tuple[EvaluationResult, ...]:
    successful = [e for e in experiments if e.status == ExperimentStatus.SUCCESS]
    if not successful:
        return ()
    perf = minmax([e.mean_score for e in successful])
    results: list[EvaluationResult] = []
    for exp, perf_n in zip(successful, perf):
        stab = stability_score(exp.std_score, exp.mean_score, exp.min_score)
        if exp.generalization_gap is None or not math.isfinite(exp.generalization_gap):
            gen = 0.5
        else:
            gen = float(max(0.0, min(1.0, 1.0 / (1.0 + abs(exp.generalization_gap)))))
        lat = latency_score(exp.infer_latency_ms, constraints.max_latency_ms if constraints else None)
        cx = _COMPLEXITY_SCORE.get(exp.complexity, 0.5)
        composite = (
            config.weight_performance * perf_n
            + config.weight_stability * stab
            + config.weight_generalization * gen
            + config.weight_latency * lat
            + config.weight_complexity * cx
        )
        failures = _constraint_failures(exp, constraints)
        results.append(
            EvaluationResult(
                model_name=exp.model_name,
                predictive=dict(exp.metrics),
                primary_score=exp.mean_score,
                stability_score=stab,
                generalization_score=gen,
                latency_ms=exp.infer_latency_ms,
                train_time_seconds=exp.train_time_seconds,
                complexity=exp.complexity,
                explainable=exp.explainable,
                constraint_failures=failures,
                passes_constraints=not failures,
                composite_score=float(composite),
                fold_scores=exp.fold_scores,
            )
        )
    return tuple(results)
