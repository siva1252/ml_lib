"""Metric computation from predictions. sklearn names stay inside this adapter."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from mlverdict.core.types import MetricPlan, ProblemDefinition


def _safe(fn, default: float = float("nan")) -> float:
    try:
        value = fn()
        return float(value)
    except Exception:
        return default


def compute_metrics(
    y_true,
    y_pred,
    y_proba,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, float] = {}
    binary = problem.is_classification and len(np.unique(y_true)) <= 2
    average = "binary" if binary else "macro"

    if problem.is_regression:
        out["mae"] = _safe(lambda: mean_absolute_error(y_true, y_pred))
        out["mse"] = _safe(lambda: mean_squared_error(y_true, y_pred))
        out["rmse"] = _safe(lambda: float(np.sqrt(mean_squared_error(y_true, y_pred))))
        out["r2"] = _safe(lambda: r2_score(y_true, y_pred))
        out["mape"] = _safe(lambda: mean_absolute_percentage_error(y_true, y_pred))
        return out

    out["accuracy"] = _safe(lambda: accuracy_score(y_true, y_pred))
    out["precision"] = _safe(lambda: precision_score(y_true, y_pred, average=average, zero_division=0))
    out["recall"] = _safe(lambda: recall_score(y_true, y_pred, average=average, zero_division=0))
    out["f1"] = _safe(lambda: f1_score(y_true, y_pred, average=average, zero_division=0))
    out["balanced_accuracy"] = _safe(lambda: balanced_accuracy_score(y_true, y_pred))

    if y_proba is not None:
        proba = np.asarray(y_proba)
        if binary:
            pos = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
            out["roc_auc"] = _safe(lambda: roc_auc_score(y_true, pos))
            out["pr_auc"] = _safe(lambda: average_precision_score(y_true, pos))
            out["log_loss"] = _safe(lambda: log_loss(y_true, proba, labels=np.unique(y_true)))
        else:
            out["roc_auc"] = _safe(lambda: roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
            out["log_loss"] = _safe(lambda: log_loss(y_true, proba, labels=np.unique(y_true)))
    return {k: v for k, v in out.items() if np.isfinite(v)}


def extract_score(metrics: dict[str, float], metric_plan: MetricPlan) -> float:
    name = metric_plan.primary.name
    if name not in metrics:
        return float("nan")
    value = metrics[name]
    if not metric_plan.primary.greater_is_better:
        return -value
    return value
