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


def _mentions(name: str, target: str) -> bool:
    """Whole-token match: 'days_since_churn' mentions 'churn'; 'days' does not mention 'y'."""
    if not target:
        return False
    tokens = re.split(r"[^a-z0-9]+", name.lower())
    return target.lower() in tokens or (len(target) >= 4 and target.lower() in name.lower())


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


def _correlation_ratio(values: pd.Series, groups: pd.Series) -> float:
    """Eta: share of a numeric column's spread explained by a categorical column (0..1)."""
    frame = pd.concat([pd.to_numeric(values, errors="coerce"), groups.astype(str)], axis=1).dropna()
    if len(frame) < 20:
        return 0.0
    v = frame.iloc[:, 0].to_numpy(dtype=float)
    total = float(((v - v.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    grouped = frame.groupby(frame.columns[1])[frame.columns[0]]
    between = float((grouped.count() * (grouped.mean() - v.mean()) ** 2).sum())
    return float(np.sqrt(max(0.0, min(1.0, between / total))))


def _binary_separation(values: pd.Series, target: pd.Series) -> float:
    """Single-feature ROC-AUC folded to 0.5..1. 1.0 means the column alone splits the classes."""
    from sklearn.metrics import roc_auc_score

    frame = pd.concat([pd.to_numeric(values, errors="coerce"), target], axis=1).dropna()
    if len(frame) < 20 or frame.iloc[:, 1].nunique() != 2:
        return 0.5
    positive = sorted(frame.iloc[:, 1].unique().tolist(), key=str)[-1]
    auc = float(roc_auc_score(frame.iloc[:, 1] == positive, frame.iloc[:, 0]))
    return max(auc, 1.0 - auc)


def detect_leakage(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    dna: DatasetDNA,
    target: str | None = None,
    *,
    group_col: str | None = None,
) -> LeakageReport:
    signals: list[LeakageSignal] = []
    has_target = bool(target) and target in frame.columns
    y = frame[target] if has_target else None
    target_l = target.lower() if has_target else ""

    for col in profile.columns:
        name = col.name
        lowered = name.lower()

        mentions_target = _mentions(name, target_l)
        if _POST_OUTCOME.search(lowered) or (mentions_target and "since" in lowered):
            # A name like days_since_last_login is a normal feature; days_since_churn is not.
            signals.append(
                _signal(
                    "post_outcome_or_future",
                    Severity.HIGH_RISK if mentions_target else Severity.WARNING,
                    f"'{name}' looks like future or post-outcome information.",
                    f"name_pattern matched on '{name}'",
                    "Confirm the timestamp relative to the label. Exclude if it is known after the outcome.",
                    name,
                )
            )
        if mentions_target and name != target:
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
        if y is None:
            continue
        try:
            feature_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
            target_numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
            target_low_card = y.nunique(dropna=True) <= 40
            if feature_numeric and target_numeric and not (target_low_card and y.nunique(dropna=True) == 2):
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
            elif feature_numeric and target_low_card and y.nunique(dropna=True) == 2:
                auc = _binary_separation(series, y)
                if auc >= 0.999:
                    signals.append(
                        _signal(
                            "perfect_target_separation",
                            Severity.HIGH_RISK,
                            f"'{name}' alone separates the two target classes almost perfectly.",
                            f"single_feature_auc={auc:.4f}",
                            "A single column this predictive is usually derived from the label. Verify before use.",
                            name,
                        )
                    )
            elif feature_numeric and target_low_card:
                eta = _correlation_ratio(series, y)
                if eta >= 0.99:
                    signals.append(
                        _signal(
                            "perfect_target_association",
                            Severity.HIGH_RISK,
                            f"'{name}' is almost fully determined by the target class.",
                            f"correlation_ratio={eta:.4f}",
                            "Inspect for a column computed from the label.",
                            name,
                        )
                    )
            elif not feature_numeric and target_numeric and not target_low_card and series.nunique(dropna=True) <= 40:
                eta = _correlation_ratio(y, series)
                if eta >= 0.99:
                    signals.append(
                        _signal(
                            "perfect_target_association",
                            Severity.HIGH_RISK,
                            f"'{name}' categories almost fully determine the target value.",
                            f"correlation_ratio={eta:.4f}",
                            "Inspect for a category derived from the label.",
                            name,
                        )
                    )
            elif series.nunique(dropna=True) <= 40 and target_low_card:
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
