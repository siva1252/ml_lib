"""Carve the locked final test first, then build a CV splitter for train/val only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from mlverdict.core.enums import ValidationStrategy
from mlverdict.core.types import ValidationPlan


@dataclass
class SplitBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    train_index: np.ndarray
    test_index: np.ndarray


def isolate_final_test(
    frame: pd.DataFrame,
    target: str,
    plan: ValidationPlan,
    random_state: int,
) -> SplitBundle:
    """Lock a final test set. Time-based uses the last slice; never reshuffle after this."""
    if len(frame) < 5:
        idx = np.arange(len(frame))
        return SplitBundle(frame.copy(), frame.iloc[0:0].copy(), idx, np.array([], dtype=int))

    if plan.time_column and plan.time_column in frame.columns:
        order = pd.to_datetime(frame[plan.time_column], errors="coerce")
        if order.notna().any():
            ordered = frame.assign(__order=order).sort_values("__order", kind="mergesort")
            n_test = max(1, int(round(len(ordered) * plan.test_size)))
            test = ordered.iloc[-n_test:].drop(columns="__order")
            train = ordered.iloc[:-n_test].drop(columns="__order")
            return SplitBundle(
                train.reset_index(drop=True),
                test.reset_index(drop=True),
                train.index.to_numpy(),
                test.index.to_numpy(),
            )

    stratify = None
    y = frame[target]
    if plan.stratify and y.nunique(dropna=True) > 1:
        counts = y.value_counts(dropna=False)
        if counts.min() >= 2:
            stratify = y

    groups = frame[plan.group_column] if plan.group_column and plan.group_column in frame.columns else None
    if groups is not None and groups.nunique() >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=plan.test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(frame, y, groups))
        return SplitBundle(
            frame.iloc[train_idx].reset_index(drop=True),
            frame.iloc[test_idx].reset_index(drop=True),
            train_idx,
            test_idx,
        )

    train, test = train_test_split(
        frame,
        test_size=plan.test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=plan.shuffle,
    )
    return SplitBundle(
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        train.index.to_numpy(),
        test.index.to_numpy(),
    )


def build_cv_splitter(plan: ValidationPlan, random_state: int, n_samples: int, n_groups: int | None = None):
    splits = min(plan.n_splits, max(2, n_samples // 4)) if n_samples else plan.n_splits
    splits = max(2, splits)
    if plan.strategy == ValidationStrategy.STRATIFIED_KFOLD:
        return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    if plan.strategy == ValidationStrategy.GROUP_KFOLD:
        n_splits = splits
        if n_groups:
            n_splits = max(2, min(splits, n_groups))
        return GroupKFold(n_splits=n_splits)
    if plan.strategy == ValidationStrategy.GROUP_SHUFFLE_SPLIT:
        return GroupShuffleSplit(n_splits=splits, test_size=plan.test_size, random_state=random_state)
    if plan.strategy == ValidationStrategy.TIME_BASED:
        return TimeSeriesSplit(n_splits=max(2, min(splits, 4)))
    if plan.strategy == ValidationStrategy.HOLDOUT:
        return ShuffleSplit(n_splits=1, test_size=plan.test_size, random_state=random_state)
    return KFold(n_splits=splits, shuffle=plan.shuffle, random_state=random_state if plan.shuffle else None)
