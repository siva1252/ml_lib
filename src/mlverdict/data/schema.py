"""Lightweight feature schema captured for the artifact."""

from __future__ import annotations

import pandas as pd

from mlverdict.core.types import FeatureSchema, PreprocessPlan


def build_feature_schema(
    frame: pd.DataFrame,
    target: str,
    plan: PreprocessPlan,
    problem_type: str,
) -> FeatureSchema:
    names = tuple(c for c in frame.columns if c != target)
    dtypes = tuple((c, str(frame[c].dtype)) for c in names)
    return FeatureSchema(
        feature_names=names,
        numeric_columns=plan.numeric_columns,
        categorical_columns=plan.categorical_columns,
        datetime_columns=plan.datetime_columns,
        target_name=target,
        problem_type=problem_type,
        dtypes=dtypes,
    )
