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

from mlverdict.core.enums import ProblemType
from mlverdict.core.types import MetricPlan, ProblemDefinition


def _safe(fn, default: float = float("nan")) -> float:
    try:
        value = fn()
        return float(value)
    except Exception:
        return default


def _positive_column(classes, positive_label) -> int:
    """Index of the positive class in predict_proba output (columns follow ``classes``)."""
    if classes is None:
        return 1
    labels = list(classes)
    if positive_label in labels:
        return labels.index(positive_label)
    return len(labels) - 1


def compute_metrics(
    y_true,
    y_pred,
    y_proba,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
    classes=None,
) -> dict[str, float]:
    """Labeled metrics. ``classes`` is the fitted model's ``classes_`` (order of probability columns).

    Binary metrics always use ``problem.positive_label`` (minority class by default), so text labels
    such as "yes"/"no" score correctly instead of silently returning NaN.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, float] = {}

    if problem.is_regression:
        out["mae"] = _safe(lambda: mean_absolute_error(y_true, y_pred))
        out["mse"] = _safe(lambda: mean_squared_error(y_true, y_pred))
        out["rmse"] = _safe(lambda: float(np.sqrt(mean_squared_error(y_true, y_pred))))
        out["r2"] = _safe(lambda: r2_score(y_true, y_pred))
        out["mape"] = _safe(lambda: mean_absolute_percentage_error(y_true, y_pred))
        return {k: v for k, v in out.items() if np.isfinite(v)}

    binary = problem.problem_type == ProblemType.BINARY_CLASSIFICATION
    out["accuracy"] = _safe(lambda: accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = _safe(lambda: balanced_accuracy_score(y_true, y_pred))

    if binary:
        pos = problem.positive_label
        if pos is None:
            pos = sorted(np.unique(y_true).tolist(), key=str)[-1]
        is_pos = y_true == pos
        pred_pos = y_pred == pos
        out["precision"] = _safe(lambda: precision_score(is_pos, pred_pos, zero_division=0))
        out["recall"] = _safe(lambda: recall_score(is_pos, pred_pos, zero_division=0))
        out["f1"] = _safe(lambda: f1_score(is_pos, pred_pos, zero_division=0))
        if y_proba is not None:
            proba = np.asarray(y_proba, dtype=float)
            if proba.ndim == 2 and proba.shape[1] > 1:
                score = proba[:, _positive_column(classes, pos)]
            else:
                score = proba.ravel()
            if is_pos.any() and (~is_pos).any():
                out["roc_auc"] = _safe(lambda: roc_auc_score(is_pos, score))
                out["pr_auc"] = _safe(lambda: average_precision_score(is_pos, score))
            labels = list(classes) if classes is not None else None
            out["log_loss"] = _safe(lambda: log_loss(y_true, proba, labels=labels))
    else:
        out["precision"] = _safe(lambda: precision_score(y_true, y_pred, average="macro", zero_division=0))
        out["recall"] = _safe(lambda: recall_score(y_true, y_pred, average="macro", zero_division=0))
        out["f1"] = _safe(lambda: f1_score(y_true, y_pred, average="macro", zero_division=0))
        if y_proba is not None:
            proba = np.asarray(y_proba, dtype=float)
            labels = list(classes) if classes is not None else None
            out["roc_auc"] = _safe(
                lambda: roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
            )
            out["log_loss"] = _safe(lambda: log_loss(y_true, proba, labels=labels))
    return {k: v for k, v in out.items() if np.isfinite(v)}


def extract_score(metrics: dict[str, float], metric_plan: MetricPlan) -> float:
    name = metric_plan.primary.name
    if name not in metrics:
        return float("nan")
    value = metrics[name]
    if not metric_plan.primary.greater_is_better:
        return -value
    return value
