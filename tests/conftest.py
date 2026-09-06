"""Shared fixtures: clean data, ugly data, and a fast Verdict config."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlverdict.core.config import VerdictConfig


@pytest.fixture
def fast_config() -> VerdictConfig:
    return VerdictConfig(
        random_state=42,
        test_size=0.25,
        cv_splits=3,
        max_hpo_trials=2,
        max_hpo_time_seconds=20,
        n_hpo_candidates=1,
        latency_probe_rows=16,
    )


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture
def binary_frame() -> pd.DataFrame:
    rng = _rng(0)
    n = 160
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    city = rng.choice(["a", "b", "c"], size=n)
    logits = 1.4 * x1 - 0.8 * x2 + (city == "a") * 0.4
    y = (logits + rng.normal(scale=0.3, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "city": city, "churn": y})


@pytest.fixture
def imbalanced_frame() -> pd.DataFrame:
    rng = _rng(1)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.zeros(n, dtype=int)
    rare = rng.choice(n, size=18, replace=False)
    y[rare] = 1
    x1 = x1 + 1.8 * y
    return pd.DataFrame({"x1": x1, "x2": x2, "churn": y})


@pytest.fixture
def multiclass_frame() -> pd.DataFrame:
    rng = _rng(2)
    n = 150
    x = rng.normal(size=(n, 3))
    y = rng.integers(0, 3, size=n)
    x[:, 0] += y
    return pd.DataFrame({"f1": x[:, 0], "f2": x[:, 1], "f3": x[:, 2], "label": y.astype(str)})


@pytest.fixture
def regression_frame() -> pd.DataFrame:
    rng = _rng(3)
    n = 140
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.5 * x1 - x2 + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "price": y})


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    rng = _rng(4)
    n = 120
    x1 = rng.normal(size=n)
    x1[:15] = np.nan
    cat = rng.choice(["red", "blue", "green"], size=n)
    const = np.ones(n)
    near = np.where(rng.random(n) < 0.97, "ok", "rare")
    row_id = np.arange(n)
    y = (np.nan_to_num(x1) + rng.normal(scale=0.2, size=n) > 0).astype(int)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "color": cat,
            "const_col": const,
            "near_const": near,
            "row_id": row_id,
            "churn": y,
        }
    )
    frame = pd.concat([frame, frame.iloc[[0, 1]]], ignore_index=True)
    return frame


@pytest.fixture
def leakage_frame() -> pd.DataFrame:
    rng = _rng(5)
    n = 100
    x1 = rng.normal(size=n)
    y = (x1 + rng.normal(scale=0.2, size=n) > 0).astype(int)
    return pd.DataFrame(
        {
            "x1": x1,
            "churn_encoded": y.astype(float),
            "days_since_churn": y * 10 + rng.normal(size=n),
            "user_id": np.arange(n),
            "churn": y,
        }
    )


@pytest.fixture
def grouped_frame() -> pd.DataFrame:
    rng = _rng(6)
    rows = []
    for customer in range(30):
        base = rng.normal()
        for _visit in range(4):
            x = base + rng.normal(scale=0.2)
            y = int(base > 0)
            rows.append({"customer_id": customer, "x": x, "churn": y})
    return pd.DataFrame(rows)


@pytest.fixture
def time_frame() -> pd.DataFrame:
    rng = _rng(7)
    n = 100
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    x = np.linspace(-1, 1, n) + rng.normal(scale=0.1, size=n)
    y = (x + np.linspace(0, 0.4, n) > 0).astype(int)
    return pd.DataFrame({"event_date": dates, "x": x, "churn": y})


@pytest.fixture
def ambiguous_target_frame() -> pd.DataFrame:
    rng = _rng(8)
    n = 80
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "score": rng.integers(1, 6, size=n).astype(float),
        }
    )
