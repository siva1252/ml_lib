"""Plan transforms from DNA + model family. Fitting happens later, on train folds only."""

from __future__ import annotations

from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import ModelFamily
from mlverdict.core.types import DatasetDNA, DatasetProfile, PreprocessPlan


def plan_preprocessing(
    profile: DatasetProfile,
    dna: DatasetDNA,
    family: ModelFamily,
    config: VerdictConfig | None = None,
    *,
    extra_drop: tuple[str, ...] = (),
) -> PreprocessPlan:
    config = config or VerdictConfig()
    drop = set(profile.constant_columns) | set(profile.identifier_columns) | set(extra_drop)
    numeric = tuple(c for c in profile.numeric_columns if c not in drop)
    categorical = tuple(c for c in profile.categorical_columns if c not in drop)
    datetime_cols = tuple(c for c in profile.datetime_columns if c not in drop)

    high_card = set(dna.high_cardinality_columns)
    if family == ModelFamily.LINEAR:
        encoding = "onehot_or_ordinal"
        scaling = True
        evidence = ("linear family: scale numeric features; one-hot low-card, ordinal high-card",)
    else:
        encoding = "ordinal"
        scaling = False
        evidence = ("tree/boosting family: ordinal categoricals, no required scaling",)

    if high_card:
        evidence = evidence + (f"high_cardinality={sorted(high_card)}",)

    return PreprocessPlan(
        numeric_imputation="median",
        categorical_imputation="most_frequent",
        numeric_scaling=scaling,
        categorical_encoding=encoding,
        datetime_features=bool(datetime_cols),
        handle_unknown="ignore",
        numeric_columns=numeric,
        categorical_columns=categorical,
        datetime_columns=datetime_cols,
        dropped_columns=tuple(sorted(drop)),
        evidence=evidence,
    )
