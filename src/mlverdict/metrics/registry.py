"""Canonical metric specs. sklearn names are adapters, not the product."""

from __future__ import annotations

from mlverdict.core.exceptions import ConfigurationError
from mlverdict.core.types import MetricSpec

CLASSIFICATION: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("accuracy", True, False, "accuracy"),
    "precision": MetricSpec("precision", True, False, "precision"),
    "recall": MetricSpec("recall", True, False, "recall"),
    "f1": MetricSpec("f1", True, False, "f1"),
    "roc_auc": MetricSpec("roc_auc", True, True, "roc_auc"),
    "pr_auc": MetricSpec("pr_auc", True, True, "average_precision"),
    "log_loss": MetricSpec("log_loss", False, True, "neg_log_loss"),
    "balanced_accuracy": MetricSpec("balanced_accuracy", True, False, "balanced_accuracy"),
}

REGRESSION: dict[str, MetricSpec] = {
    "mae": MetricSpec("mae", False, False, "neg_mean_absolute_error"),
    "mse": MetricSpec("mse", False, False, "neg_mean_squared_error"),
    "rmse": MetricSpec("rmse", False, False, "neg_root_mean_squared_error"),
    "r2": MetricSpec("r2", True, False, "r2"),
    "mape": MetricSpec("mape", False, False, "neg_mean_absolute_percentage_error"),
}

ALL = {**CLASSIFICATION, **REGRESSION}


def get_metric(name: str) -> MetricSpec:
    key = name.lower()
    aliases = {"average_precision": "pr_auc", "msle": "mse", "neg_log_loss": "log_loss"}
    key = aliases.get(key, key)
    if key not in ALL:
        raise ConfigurationError(f"Unknown metric '{name}'. Known: {sorted(ALL)}")
    return ALL[key]
