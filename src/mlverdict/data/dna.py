"""Compress a DatasetProfile into machine-readable Dataset DNA."""

from __future__ import annotations

from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import ColumnRole, DatasetScale, IIDAssumption
from mlverdict.core.types import SCHEMA_VERSION, DatasetDNA, DatasetProfile


def _scale(n_rows: int) -> DatasetScale:
    if n_rows < 200:
        return DatasetScale.TINY
    if n_rows < 2000:
        return DatasetScale.SMALL
    if n_rows < 50000:
        return DatasetScale.MEDIUM
    return DatasetScale.LARGE


def _missingness(profile: DatasetProfile) -> str:
    if not profile.columns:
        return "none"
    ratios = [c.missing_ratio for c in profile.columns]
    avg = sum(ratios) / len(ratios)
    if avg <= 0.001:
        return "none"
    if avg < 0.05:
        return "low"
    if avg < 0.20:
        return "moderate"
    return "high"


def _target_kind(profile: DatasetProfile) -> str:
    if not profile.target.name:
        return "none"
    dtype = profile.target.dtype
    if dtype.startswith("float") and profile.target.n_unique > 20:
        return "numeric"
    if profile.target.n_unique <= 20:
        return "categorical"
    if dtype.startswith("float") or dtype.startswith("int"):
        return "numeric"
    return "categorical"


def _entity_hint(profile: DatasetProfile, group_col: str | None) -> str | None:
    if group_col:
        return group_col
    for col in profile.columns:
        name = col.name.lower()
        repeated_entity = (
            name.endswith("_id") or name in {"customer_id", "user_id", "account_id", "entity_id"}
        ) and col.unique_ratio < 0.9 and col.n_unique > 1
        if repeated_entity:
            return col.name
    return None


def _time_hint(profile: DatasetProfile, time_col: str | None) -> str | None:
    if time_col:
        return time_col
    if profile.datetime_columns:
        return profile.datetime_columns[0]
    return None


def _iid(entity: str | None, time: str | None, group_confirmed: bool, time_confirmed: bool) -> IIDAssumption:
    if group_confirmed or time_confirmed:
        return IIDAssumption.UNLIKELY
    if entity or time:
        return IIDAssumption.DOUBTFUL
    return IIDAssumption.LIKELY


def build_dna(
    profile: DatasetProfile,
    config: VerdictConfig | None = None,
    *,
    group_col: str | None = None,
    time_col: str | None = None,
) -> DatasetDNA:
    config = config or VerdictConfig()
    n_features = len(profile.columns)
    high_card = tuple(
        c.name
        for c in profile.columns
        if c.role == ColumnRole.CATEGORICAL and c.n_unique >= config.high_cardinality_threshold
    )
    near_constant = tuple(c.name for c in profile.columns if c.is_near_constant)
    entity = _entity_hint(profile, group_col)
    time = _time_hint(profile, time_col)
    numeric_n = len(profile.numeric_columns)
    categorical_n = len(profile.categorical_columns)
    return DatasetDNA(
        schema_version=SCHEMA_VERSION,
        n_rows=profile.n_rows,
        n_features=n_features,
        target_name=profile.target.name,
        target_kind=_target_kind(profile),
        target_cardinality=profile.target.n_unique,
        imbalance_ratio=profile.target.imbalance_ratio,
        missingness=_missingness(profile),
        has_duplicates=profile.n_duplicate_rows > 0,
        has_datetime=bool(profile.datetime_columns) or time_col is not None,
        has_potential_ids=bool(profile.identifier_columns),
        potential_id_columns=profile.identifier_columns,
        high_cardinality_columns=high_card,
        constant_columns=profile.constant_columns,
        near_constant_columns=near_constant,
        entity_hint=entity,
        time_hint=time,
        iid_assumption=_iid(entity, time, group_col is not None, time_col is not None),
        scale=_scale(profile.n_rows),
        numeric_fraction=numeric_n / max(n_features, 1),
        categorical_fraction=categorical_n / max(n_features, 1),
    )
