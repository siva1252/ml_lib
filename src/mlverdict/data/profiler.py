"""Dataset profiler. Operates on the train/val world, never the locked final test."""

from __future__ import annotations

import pandas as pd

from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import ColumnRole
from mlverdict.core.types import SCHEMA_VERSION, ColumnProfile, DatasetProfile, TargetProfile

_ID_NAME_HINTS = (
    "id",
    "uuid",
    "guid",
    "ssn",
    "customer_id",
    "user_id",
    "account_id",
    "entity_id",
    "row_id",
)
_TIME_NAME_HINTS = ("date", "time", "timestamp", "_at", "_ts", "datetime")


def _is_datetime(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    lowered = name.lower()
    if any(h in lowered for h in _TIME_NAME_HINTS):
        sample = series.dropna().astype(str).head(20)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean()) >= 0.8
    return False


def _role(series: pd.Series, name: str, *, n_rows: int, config: VerdictConfig) -> ColumnRole:
    n_unique = int(series.nunique(dropna=True))
    if n_unique <= 1:
        return ColumnRole.CONSTANT
    if _is_datetime(series, name):
        return ColumnRole.DATETIME
    unique_ratio = n_unique / max(n_rows, 1)
    lowered = name.lower()
    id_name = lowered in _ID_NAME_HINTS or lowered.endswith("_id")
    if id_name and unique_ratio >= config.id_unique_ratio:
        return ColumnRole.IDENTIFIER
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if n_unique <= 12 and unique_ratio < 0.05:
            return ColumnRole.CATEGORICAL
        return ColumnRole.NUMERIC
    if pd.api.types.is_bool_dtype(series):
        return ColumnRole.CATEGORICAL
    avg_len = series.dropna().astype(str).head(50).map(len).mean() if series.notna().any() else 0
    if avg_len and avg_len > 80:
        return ColumnRole.TEXT
    return ColumnRole.CATEGORICAL


def _column_profile(series: pd.Series, name: str, config: VerdictConfig) -> ColumnProfile:
    n_rows = len(series)
    n_unique = int(series.nunique(dropna=True))
    n_missing = int(series.isna().sum())
    unique_ratio = n_unique / max(n_rows, 1)
    missing_ratio = n_missing / max(n_rows, 1)
    role = _role(series, name, n_rows=n_rows, config=config)
    is_constant = n_unique <= 1
    is_near_constant = False
    if not is_constant and n_rows:
        top = series.value_counts(dropna=True, normalize=True)
        if not top.empty and float(top.iloc[0]) >= config.near_constant_threshold:
            is_near_constant = True
    lowered = name.lower()
    name_hint = lowered in _ID_NAME_HINTS or lowered.endswith("_id")
    integer_like = False
    if role != ColumnRole.DATETIME and pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if not clean.empty:
            integer_like = bool(((clean % 1) == 0).all())
    # Continuous floats are almost unique by nature — that is not an identifier.
    is_potential_id = role == ColumnRole.IDENTIFIER or (
        (name_hint or integer_like)
        and unique_ratio >= config.id_unique_ratio
        and n_unique >= max(20, int(0.9 * n_rows))
        and role != ColumnRole.DATETIME
    )
    sample = tuple(series.dropna().astype(str).head(3).tolist())
    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        role=role,
        n_unique=n_unique,
        unique_ratio=float(unique_ratio),
        n_missing=n_missing,
        missing_ratio=float(missing_ratio),
        is_constant=is_constant,
        is_near_constant=is_near_constant,
        is_potential_id=is_potential_id,
        sample_values=sample,
    )


def _target_profile(series: pd.Series, name: str) -> TargetProfile:
    clean = series.dropna()
    n_unique = int(clean.nunique())
    n_missing = int(series.isna().sum())
    unique_ratio = n_unique / max(len(series), 1)
    class_counts = None
    imbalance_ratio = None
    mean = std = min_value = max_value = None
    numeric = pd.api.types.is_numeric_dtype(clean) and not pd.api.types.is_bool_dtype(clean)
    if n_unique <= 30 or not numeric:
        counts = clean.astype(str).value_counts()
        class_counts = tuple((str(k), int(v)) for k, v in counts.items())
        if len(counts) >= 2:
            imbalance_ratio = float(counts.iloc[0] / max(counts.iloc[-1], 1))
    if numeric and not clean.empty:
        mean = float(clean.mean())
        std = float(clean.std(ddof=0)) if len(clean) else 0.0
        min_value = float(clean.min())
        max_value = float(clean.max())
    return TargetProfile(
        name=name,
        dtype=str(series.dtype),
        n_unique=n_unique,
        unique_ratio=float(unique_ratio),
        n_missing=n_missing,
        class_counts=class_counts,
        imbalance_ratio=imbalance_ratio,
        mean=mean,
        std=std,
        min_value=min_value,
        max_value=max_value,
    )


def profile_dataset(
    frame: pd.DataFrame,
    target: str | None,
    config: VerdictConfig | None = None,
) -> DatasetProfile:
    config = config or VerdictConfig()
    columns = []
    for name in frame.columns:
        if target and name == target:
            continue
        columns.append(_column_profile(frame[name], name, config))
    cols = tuple(columns)
    if target and target in frame.columns:
        target_prof = _target_profile(frame[target], target)
    else:
        target_prof = TargetProfile(
            name="",
            dtype="none",
            n_unique=0,
            unique_ratio=0.0,
            n_missing=0,
        )
    return DatasetProfile(
        schema_version=SCHEMA_VERSION,
        n_rows=int(len(frame)),
        n_columns=int(frame.shape[1]),
        n_duplicate_rows=int(frame.duplicated().sum()),
        columns=cols,
        target=target_prof,
        numeric_columns=tuple(c.name for c in cols if c.role == ColumnRole.NUMERIC),
        categorical_columns=tuple(c.name for c in cols if c.role == ColumnRole.CATEGORICAL),
        datetime_columns=tuple(c.name for c in cols if c.role == ColumnRole.DATETIME),
        identifier_columns=tuple(c.name for c in cols if c.role == ColumnRole.IDENTIFIER or c.is_potential_id),
        constant_columns=tuple(c.name for c in cols if c.is_constant),
    )
