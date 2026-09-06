"""Catalog of candidate families. Availability is resolved at selection time."""

from __future__ import annotations

from mlverdict.core.enums import ModelFamily
from mlverdict.models.adapters import available_boosters

CATALOG = (
    {
        "name": "Logistic Regression",
        "key": "logistic_regression",
        "family": ModelFamily.LINEAR,
        "tasks": ("classification",),
        "explainable": True,
        "complexity": "low",
        "optional": False,
    },
    {
        "name": "Ridge",
        "key": "ridge",
        "family": ModelFamily.LINEAR,
        "tasks": ("regression",),
        "explainable": True,
        "complexity": "low",
        "optional": False,
    },
    {
        "name": "Random Forest",
        "key": "random_forest",
        "family": ModelFamily.TREE,
        "tasks": ("classification", "regression"),
        "explainable": True,
        "complexity": "medium",
        "optional": False,
    },
    {
        "name": "Extra Trees",
        "key": "extra_trees",
        "family": ModelFamily.TREE,
        "tasks": ("classification", "regression"),
        "explainable": True,
        "complexity": "medium",
        "optional": False,
    },
    {
        "name": "HistGradientBoosting",
        "key": "hist_gradient_boosting",
        "family": ModelFamily.BOOSTING,
        "tasks": ("classification", "regression"),
        "explainable": False,
        "complexity": "high",
        "optional": False,
    },
    {
        "name": "XGBoost",
        "key": "xgboost",
        "family": ModelFamily.BOOSTING,
        "tasks": ("classification", "regression"),
        "explainable": False,
        "complexity": "high",
        "optional": True,
    },
    {
        "name": "LightGBM",
        "key": "lightgbm",
        "family": ModelFamily.BOOSTING,
        "tasks": ("classification", "regression"),
        "explainable": False,
        "complexity": "high",
        "optional": True,
    },
    {
        "name": "CatBoost",
        "key": "catboost",
        "family": ModelFamily.BOOSTING,
        "tasks": ("classification", "regression"),
        "explainable": False,
        "complexity": "high",
        "optional": True,
    },
)


def is_available(key: str) -> bool:
    boosters = available_boosters()
    if key in boosters:
        return boosters[key]
    return True
