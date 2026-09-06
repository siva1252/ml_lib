"""Leakage *signal* detector. Never claims a dataset is leakage-free."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from mlverdict.core.enums import Severity
from mlverdict.core.types import DatasetDNA, DatasetProfile, LeakageReport, LeakageSignal


_POST_OUTCOME = re.compile(
    r"(after_|post_|future_|days_since_|hours_since_|weeks_since_|_after$|_post$)",
    re.I,
)
_TARGET_DERIVED = re.compile(
    r"(_label|_target|_encoded|_leak|actual_|ground_truth|true_)",
    re.I,
)


def _signal(
    signal_type: str,
    severity: Severity,
    description: str,
    evidence: str,
    recommendation: str,
    column: str | None = None,
) -> LeakageSignal:
    return LeakageSignal(
        signal_type=signal_type,
        severity=severity,
        column=column,
        evidence=evidence,
        description=description,
        recommendation=recommendation,
    )


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = 0.0
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0
    expected = table.to_numpy().sum(axis=1, keepdims=True) * table.to_numpy().sum(axis=0, keepdims=True) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((table.to_numpy() - expected) ** 2 / np.where(expected == 0, np.nan, expected)))
    r, k = table.shape
    denom = n * (min(r, k) - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def detect_leakage(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    dna: DatasetDNA,
    target: str,
    *,
    group_col: str | None = None,
) -> LeakageReport:
    signals: list[LeakageSignal] = []
    y = frame[target]
    target_l = target.lower()

    for col in profile.columns:
        name = col.name
        lowered = name.lower()

        if _POST_OUTCOME.search(lowered) or f"since_{target_l}" in lowered:
            signals.append(
                _signal(
                    "post_outcome_or_future",
                    Severity.HIGH_RISK,
                    f"'{name}' looks like future or post-outcome information.",
                    f"name_pattern matched on '{name}'",
                    "Confirm the timestamp relative to the label. Exclude if it is known after the outcome.",
                    name,
                )
            )
        if target_l and target_l in lowered and name != target:
            signals.append(
                _signal(
                    "target_derived_name",
                    Severity.HIGH_RISK,
                    f"'{name}' embeds the target name and may be derived from the label.",
                    f"column contains '{target}'",
                    "Inspect how the column was created. Exclude target-derived features.",
                    name,
                )
            )
        elif _TARGET_DERIVED.search(lowered):
            signals.append(
                _signal(
                    "target_derived_name",
                    Severity.WARNING,
                    f"'{name}' uses a label-like suffix.",
                    f"name_pattern matched on '{name}'",
                    "Verify it is not computed from the target.",
                    name,
                )
            )

        if col.is_potential_id:
            signals.append(
                _signal(
                    "id_leakage",
                    Severity.HIGH_RISK,
                    f"'{name}' may leak identity rather than a generalizable pattern.",
                    f"unique_ratio={col.unique_ratio:.4f}",
                    "Drop as a feature. If it identifies a group, pass group_col and use grouped validation.",
                    name,
                )
            )

        series = frame[name]
        try:
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(y):
                aligned = pd.concat([series, y], axis=1).dropna()
                if len(aligned) >= 20:
                    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                    if np.isfinite(corr) and abs(corr) >= 0.999:
                        signals.append(
                            _signal(
                                "perfect_target_correlation",
                                Severity.HIGH_RISK,
                                f"'{name}' is almost perfectly correlated with the target.",
                                f"pearson={corr:.6f}",
                                "Treat as a likely target-derived or post-outcome feature until proven otherwise.",
                                name,
                            )
                        )
            elif series.nunique(dropna=True) <= 40 and y.nunique(dropna=True) <= 40:
                v = _cramers_v(series.astype(str), y.astype(str))
                if v >= 0.99:
                    signals.append(
                        _signal(
                            "perfect_target_association",
                            Severity.HIGH_RISK,
                            f"'{name}' is almost perfectly associated with the target.",
                            f"cramers_v={v:.4f}",
                            "Inspect for target encoding fitted on the full dataset.",
                            name,
                        )
                    )
        except Exception:
            pass

    if dna.has_duplicates:
        signals.append(
            _signal(
                "duplicate_entities",
                Severity.WARNING,
                "Duplicate rows can leak the same entity into train and validation.",
                f"duplicate_rows present; iid={dna.iid_assumption.value}",
                "Deduplicate or validate by group.",
            )
        )

    if group_col is None and dna.entity_hint:
        signals.append(
            _signal(
                "ungrouped_repeated_entity",
                Severity.WARNING,
                f"Repeated entity hint '{dna.entity_hint}' without group_col.",
                "IID assumption is doubtful; random splits may contaminate folds.",
                f"Pass group_col='{dna.entity_hint}' if this is the same entity across rows.",
                dna.entity_hint,
            )
        )

    signals.append(
        _signal(
            "preprocessing_leakage_risk",
            Severity.INFO,
            "Any transform fitted on the full dataset (or the final test) is a leakage path.",
            "This engine fits preprocessing only on training folds.",
            "Keep the preprocess+model pipeline inside cross-validation.",
        )
    )

    return LeakageReport(signals=tuple(signals))
