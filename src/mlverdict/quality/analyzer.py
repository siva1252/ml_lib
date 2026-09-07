"""Data-quality engine. Issues are recorded; silent cleanup is not quality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import Severity
from mlverdict.core.types import DatasetProfile, QualityIssue, QualityReport
from mlverdict.quality.issues import issue


def analyze_quality(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    config: VerdictConfig | None = None,
) -> QualityReport:
    config = config or VerdictConfig()
    issues: list[QualityIssue] = []
    target = profile.target

    if target.name and target.n_missing:
        sev = Severity.BLOCKER if target.n_missing == profile.n_rows else Severity.HIGH_RISK
        issues.append(
            issue(
                "missing_target",
                sev,
                "Target contains missing values.",
                f"missing={target.n_missing}/{profile.n_rows}",
                "Drop or impute target rows before modeling; missing labels cannot be trained on.",
                target.name,
            )
        )

    if profile.n_rows < 10:
        issues.append(
            issue(
                "too_few_rows",
                Severity.BLOCKER,
                "Dataset is too small for a reliable experiment.",
                f"n_rows={profile.n_rows}",
                "Collect more rows.",
            )
        )

    usable = [
        c
        for c in profile.columns
        if not c.is_constant and not c.is_potential_id
    ]
    if not usable:
        issues.append(
            issue(
                "no_usable_features",
                Severity.BLOCKER,
                "Every feature is constant or identifier-like.",
                f"n_columns={len(profile.columns)}",
                "Provide predictive features that vary across rows.",
            )
        )

    if profile.n_duplicate_rows:
        sev = Severity.HIGH_RISK if profile.n_duplicate_rows / max(profile.n_rows, 1) > 0.1 else Severity.WARNING
        issues.append(
            issue(
                "duplicate_rows",
                sev,
                "Duplicate rows can inflate validation scores.",
                f"duplicates={profile.n_duplicate_rows}",
                "Deduplicate before training or use grouped validation if they are the same entity.",
            )
        )

    for col in profile.columns:
        if col.n_missing:
            ratio = col.missing_ratio
            if ratio >= 0.9:
                sev = Severity.HIGH_RISK
            elif ratio >= 0.4:
                sev = Severity.HIGH_RISK
            elif ratio >= 0.05:
                sev = Severity.WARNING
            else:
                sev = Severity.INFO
            issues.append(
                issue(
                    "missing_values",
                    sev,
                    f"Column '{col.name}' has missing values.",
                    f"missing_ratio={ratio:.3f}",
                    "Impute inside the training fold only; do not fill using the full dataset.",
                    col.name,
                )
            )
        if col.is_constant:
            issues.append(
                issue(
                    "constant_feature",
                    Severity.WARNING,
                    f"Column '{col.name}' is constant and has no predictive value.",
                    f"n_unique={col.n_unique}",
                    "Drop the column from the feature set.",
                    col.name,
                )
            )
        elif col.is_near_constant:
            issues.append(
                issue(
                    "near_constant_feature",
                    Severity.WARNING,
                    f"Column '{col.name}' is near-constant.",
                    f"unique_ratio={col.unique_ratio:.4f}",
                    "Consider dropping it; it rarely changes.",
                    col.name,
                )
            )
        if col.is_potential_id:
            issues.append(
                issue(
                    "potential_identifier",
                    Severity.HIGH_RISK,
                    f"Column '{col.name}' looks like an identifier.",
                    f"unique_ratio={col.unique_ratio:.4f} n_unique={col.n_unique}",
                    "Exclude from features unless it is a known grouping key (then pass group_col).",
                    col.name,
                )
            )
        if col.n_unique >= config.high_cardinality_threshold and col.role.value == "categorical":
            issues.append(
                issue(
                    "high_cardinality",
                    Severity.WARNING,
                    f"Column '{col.name}' has high cardinality.",
                    f"n_unique={col.n_unique}",
                    "Prefer ordinal/target encoding or grouping rare levels; avoid dense one-hot.",
                    col.name,
                )
            )
        series = frame[col.name]
        if pd.api.types.is_numeric_dtype(series):
            n_inf = int(np.isinf(pd.to_numeric(series, errors="coerce")).sum())
            if n_inf:
                issues.append(
                    issue(
                        "invalid_values",
                        Severity.HIGH_RISK,
                        f"Column '{col.name}' contains infinities.",
                        f"n_inf={n_inf}",
                        "Replace infinities with NaN and impute inside the training pipeline.",
                        col.name,
                    )
                )

    return QualityReport(issues=tuple(issues))
