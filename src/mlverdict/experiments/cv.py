"""Cross-validation on the train/val world only. Records fold stats and train/val gap."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from mlverdict.core.enums import ExperimentStatus, ValidationStrategy
from mlverdict.core.types import (
    CandidateModel,
    ExperimentResult,
    MetricPlan,
    ProblemDefinition,
    ValidationPlan,
)
from mlverdict.experiments.scoring import compute_metrics, extract_score
from mlverdict.validation.splitter import build_cv_splitter


def _predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            return None
    return None


def _latency_ms(model, X, n_probe: int) -> float:
    if len(X) == 0:
        return 0.0
    probe = X.iloc[: min(n_probe, len(X))]
    start = time.perf_counter()
    model.predict(probe)
    elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0)


def run_cross_validation(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    candidate: CandidateModel,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
    validation: ValidationPlan,
    *,
    random_state: int,
    groups: pd.Series | None = None,
    latency_probe_rows: int = 64,
    configuration: dict[str, Any] | None = None,
    optimized: bool = False,
) -> ExperimentResult:
    n_groups = int(groups.nunique()) if groups is not None else None
    splitter = build_cv_splitter(validation, random_state, len(X), n_groups)
    fold_scores: list[float] = []
    train_scores: list[float] = []
    metric_accum: dict[str, list[float]] = {}
    last_model = None
    last_val_x = X
    t0 = time.perf_counter()
    error = None
    status = ExperimentStatus.SUCCESS

    try:
        y_arr = y.reset_index(drop=True)
        X_arr = X.reset_index(drop=True)
        g_arr = groups.reset_index(drop=True) if groups is not None else None
        split_kwargs = {}
        if validation.strategy in {ValidationStrategy.GROUP_KFOLD, ValidationStrategy.GROUP_SHUFFLE_SPLIT}:
            if g_arr is None:
                raise ValueError("Grouped validation requires groups.")
            splits = splitter.split(X_arr, y_arr, g_arr)
        else:
            splits = splitter.split(X_arr, y_arr, **split_kwargs)

        for train_idx, val_idx in splits:
            model = clone(pipeline)
            X_tr, X_va = X_arr.iloc[train_idx], X_arr.iloc[val_idx]
            y_tr, y_va = y_arr.iloc[train_idx], y_arr.iloc[val_idx]
            model.fit(X_tr, y_tr)
            pred_va = model.predict(X_va)
            proba_va = _predict_proba(model, X_va)
            metrics_va = compute_metrics(y_va, pred_va, proba_va, problem, metric_plan)
            fold_scores.append(extract_score(metrics_va, metric_plan))
            for key, value in metrics_va.items():
                metric_accum.setdefault(key, []).append(value)
            pred_tr = model.predict(X_tr)
            proba_tr = _predict_proba(model, X_tr)
            metrics_tr = compute_metrics(y_tr, pred_tr, proba_tr, problem, metric_plan)
            train_scores.append(extract_score(metrics_tr, metric_plan))
            last_model = model
            last_val_x = X_va
    except Exception as exc:
        status = ExperimentStatus.FAILED
        error = str(exc)
        fold_scores = [float("nan")]

    train_time = time.perf_counter() - t0
    scores = np.asarray(fold_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    mean = float(scores.mean()) if len(scores) else float("nan")
    std = float(scores.std(ddof=0)) if len(scores) else float("nan")
    mn = float(scores.min()) if len(scores) else float("nan")
    mx = float(scores.max()) if len(scores) else float("nan")
    train_mean = float(np.nanmean(train_scores)) if train_scores else None
    gap = None
    if train_mean is not None and np.isfinite(mean) and np.isfinite(train_mean):
        gap = float(train_mean - mean)
    latency = _latency_ms(last_model, last_val_x, latency_probe_rows) if last_model is not None else float("inf")
    agg_metrics = {k: float(np.nanmean(v)) for k, v in metric_accum.items()}
    if metric_plan.primary.name not in agg_metrics and np.isfinite(mean):
        raw = mean if metric_plan.primary.greater_is_better else -mean
        agg_metrics[metric_plan.primary.name] = float(raw)

    return ExperimentResult(
        model_name=candidate.name,
        estimator_key=candidate.estimator_key,
        configuration=configuration or dict(candidate.params),
        metrics=agg_metrics,
        fold_scores=tuple(float(s) for s in fold_scores),
        mean_score=mean,
        std_score=std if np.isfinite(std) else 0.0,
        min_score=mn,
        max_score=mx,
        train_score=train_mean,
        generalization_gap=gap,
        train_time_seconds=float(train_time),
        infer_latency_ms=float(latency),
        status=status,
        error=error,
        n_samples=len(X),
        n_features=X.shape[1],
        validation_strategy=validation.strategy.value,
        optimized=optimized,
        family=candidate.family.value,
        explainable=candidate.explainable,
        complexity=candidate.complexity,
    )
