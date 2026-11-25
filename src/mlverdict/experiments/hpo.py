"""Bounded Optuna HPO on promising candidates only. No unlimited search."""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd

from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import ExperimentStatus, ModelFamily
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

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _space(trial: optuna.Trial, key: str) -> dict[str, Any]:
    if key in {"logistic_regression"}:
        return {"C": trial.suggest_float("C", 1e-2, 10.0, log=True)}
    if key == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 20.0, log=True)}
    if key == "lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}
    if key == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    if key == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    if key == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    if key == "svm":
        return {"C": trial.suggest_float("C", 1e-2, 100.0, log=True)}
    if key == "gaussian_nb":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-11, 1e-5, log=True)}
    if key in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 40, 140),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
    if key == "hist_gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "max_iter": trial.suggest_int("max_iter", 40, 120),
        }
    if key in {"xgboost", "lightgbm", "catboost"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 40, 140),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        }
    return {}


def optimize_candidate(
    candidate: CandidateModel,
    X: pd.DataFrame,
    y: pd.Series,
    profile: DatasetProfile,
    dna: DatasetDNA,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
    validation: ValidationPlan,
    config: VerdictConfig,
    *,
    groups: pd.Series | None = None,
    extra_drop: tuple[str, ...] = (),
) -> ExperimentResult:
    assert problem.problem_type is not None
    family = candidate.family if isinstance(candidate.family, ModelFamily) else ModelFamily(candidate.family)
    plan = plan_preprocessing(profile, dna, family, config, extra_drop=extra_drop)

    if not has_search_space(_space, candidate.estimator_key):
        # Nothing to tune (e.g. plain LinearRegression): one CV run with defaults, not N identical trials.
        pipeline = build_pipeline(plan, candidate, problem.problem_type, config.random_state, {})
        return run_cross_validation(
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
            configuration={},
            optimized=False,
        )

    def objective(trial: optuna.Trial) -> float:
        params = _space(trial, candidate.estimator_key)
        pipeline = build_pipeline(plan, candidate, problem.problem_type, config.random_state, params)
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
            configuration=params,
            optimized=True,
        )
        if result.status != ExperimentStatus.SUCCESS or not np_finite(result.mean_score):
            raise optuna.TrialPruned()
        return result.mean_score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
    )
    study.optimize(
        objective,
        n_trials=config.max_hpo_trials,
        timeout=config.max_hpo_time_seconds,
        catch=(Exception,),
    )
    best_params = best_params_or_default(study)
    pipeline = build_pipeline(plan, candidate, problem.problem_type, config.random_state, best_params)
    return run_cross_validation(
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
        configuration=best_params,
        optimized=True,
    )


def has_search_space(space_fn, key: str) -> bool:
    """True if ``space_fn`` asks the trial for at least one parameter for this model key."""
    try:
        return bool(space_fn(optuna.trial.FixedTrial({}), key))
    except Exception:
        # FixedTrial({}) raises as soon as a parameter is suggested; any other error is left
        # to the study, which records failed trials instead of crashing.
        return True


def best_params_or_default(study: optuna.Study) -> dict[str, Any]:
    """Best trial's params, or {} (library defaults) when no trial completed.

    ``study.best_trial`` raises ValueError when every trial failed or was pruned,
    which used to crash the whole run.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return {}
    return dict(max(completed, key=lambda t: t.value).params)


def np_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def select_promising(
    results: tuple[ExperimentResult, ...],
    n: int,
) -> tuple[ExperimentResult, ...]:
    ok = [r for r in results if r.status == ExperimentStatus.SUCCESS and np_finite(r.mean_score)]
    ok.sort(key=lambda r: (r.mean_score, -r.std_score), reverse=True)
    return tuple(ok[: max(1, n)]) if ok else ()
