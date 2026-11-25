"""Load a tabular dataset from a path or an in-memory DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlverdict.core.exceptions import DataLoadError, TargetNotFoundError


def _check_columns(frame: pd.DataFrame) -> pd.DataFrame:
    # Column names are always strings inside MLVerdict (pd.DataFrame(array) gives 0, 1, 2, ...).
    frame.columns = [str(c) for c in frame.columns]
    dupes = frame.columns[frame.columns.duplicated()].unique().tolist()
    if dupes:
        raise DataLoadError(f"Duplicate column names: {dupes}. Rename or drop them before fitting.")
    return frame


def load_dataset(data: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise DataLoadError("DataFrame is empty.")
        return _check_columns(data.copy())

    path = Path(data)
    if not path.exists():
        raise DataLoadError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix in {".json"}:
            frame = pd.read_json(path)
        else:
            raise DataLoadError(f"Unsupported file type: {suffix or 'none'}")
    except DataLoadError:
        raise
    except Exception as exc:  # pragma: no cover - pandas IO errors vary
        raise DataLoadError(f"Failed to load {path}: {exc}") from exc

    if frame.empty:
        raise DataLoadError(f"Loaded dataset is empty: {path}")
    return _check_columns(frame)


def require_target(frame: pd.DataFrame, target: str) -> None:
    if target not in frame.columns:
        raise TargetNotFoundError(
            f"Target '{target}' is not in columns: {list(frame.columns)}"
        )
    if frame[target].dropna().empty:
        raise TargetNotFoundError(f"Target '{target}' has no non-null values.")
