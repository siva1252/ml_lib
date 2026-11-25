"""Detect the ML problem from target evidence. Never silently force a low-confidence call."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from mlverdict.core.enums import Confidence, DecisionStatus, ProblemType, UnsupervisedTask
from mlverdict.core.exceptions import ConfigurationError
from mlverdict.core.types import DatasetProfile, ProblemDefinition

_SUPERVISED_OVERRIDES = {
    ProblemType.BINARY_CLASSIFICATION.value: ProblemType.BINARY_CLASSIFICATION,
    ProblemType.MULTICLASS_CLASSIFICATION.value: ProblemType.MULTICLASS_CLASSIFICATION,
    ProblemType.REGRESSION.value: ProblemType.REGRESSION,
}
_UNSUPERVISED_FROM_TASK = {
    UnsupervisedTask.CLUSTERING: ProblemType.CLUSTERING,
    UnsupervisedTask.ANOMALY_DETECTION: ProblemType.ANOMALY_DETECTION,
    UnsupervisedTask.DIMENSIONALITY_REDUCTION: ProblemType.DIMENSIONALITY_REDUCTION,
}


def unsupervised_problem(task: UnsupervisedTask) -> ProblemDefinition:
    return ProblemDefinition(
        problem_type=_UNSUPERVISED_FROM_TASK[task],
        confidence=Confidence.HIGH,
        evidence=(f"User selected unsupervised task={task.value}",),
        warnings=("No labeled target; scores are internal structure metrics, not labeled accuracy.",),
        status=DecisionStatus.DECIDED,
        user_override=True,
    )


def _looks_integer_like(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    clean = series.dropna()
    if clean.empty:
        return False
    return bool(((clean * 10).round() / 10 == clean.round()).mean() > 0.98)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def minority_label(target: pd.Series) -> Any:
    """The rarer class; ties go to the larger sorted label (so 0/1 targets keep 1 as positive)."""
    counts = target.dropna().value_counts()
    if counts.empty:
        return None
    labels = sorted(counts.index.tolist(), key=lambda v: (str(type(v)), v))
    rarest = min(counts.loc[lbl] for lbl in labels)
    return _python_scalar([lbl for lbl in labels if counts.loc[lbl] == rarest][-1])


def _resolve_positive(target: pd.Series, user_positive: Any) -> tuple[Any, str]:
    if user_positive is not None:
        present = set(target.dropna().unique().tolist())
        if user_positive not in present:
            raise ConfigurationError(
                f"positive_label={user_positive!r} is not a value of the target. Present: {sorted(map(str, present))}"
            )
        return _python_scalar(user_positive), f"positive_label={user_positive!r} (user)"
    label = minority_label(target)
    return label, f"positive_label={label!r} (minority class)"


def _is_class_code(target: pd.Series, n_unique: int) -> bool:
    """0..k-1 integer codes (the usual encoding of class labels), not a float rating scale."""
    if not pd.api.types.is_integer_dtype(target):
        return False
    values = sorted(int(v) for v in target.dropna().unique())
    return values == list(range(n_unique))


def detect_problem(
    profile: DatasetProfile,
    target: pd.Series,
    *,
    user_problem_type: str | ProblemType | None = None,
    positive_label: Any = None,
) -> ProblemDefinition:
    problem = _detect(profile, target, user_problem_type=user_problem_type)
    if problem.problem_type == ProblemType.BINARY_CLASSIFICATION:
        label, note = _resolve_positive(target, positive_label)
        return replace(problem, positive_label=label, evidence=problem.evidence + (note,))
    if positive_label is not None and problem.problem_type is not None:
        raise ConfigurationError("positive_label only applies to binary classification.")
    return problem


def _detect(
    profile: DatasetProfile,
    target: pd.Series,
    *,
    user_problem_type: str | ProblemType | None = None,
) -> ProblemDefinition:
    if user_problem_type is not None:
        key = user_problem_type.value if isinstance(user_problem_type, ProblemType) else str(user_problem_type)
        if key in {item.value for item in UnsupervisedTask}:
            raise ConfigurationError(
                "Unsupervised objectives use task= without a target, not problem_type with a labeled column."
            )
        if key not in _SUPERVISED_OVERRIDES:
            raise ConfigurationError(f"Unknown problem_type '{key}'.")
        return ProblemDefinition(
            problem_type=_SUPERVISED_OVERRIDES[key],
            confidence=Confidence.HIGH,
            evidence=(f"User override: problem_type={key}",),
            warnings=("Problem type was set by the user; automatic evidence was not used for the decision.",),
            status=DecisionStatus.DECIDED,
            user_override=True,
        )

    n_unique = profile.target.n_unique
    dtype = profile.target.dtype
    numeric = dtype.startswith("int") or dtype.startswith("float") or dtype == "int64" or dtype == "float64"
    bool_like = dtype in {"bool", "boolean"}
    evidence: list[str] = [
        f"target_dtype={dtype}",
        f"target_cardinality={n_unique}",
        f"unique_ratio={profile.target.unique_ratio:.4f}",
    ]
    if profile.target.imbalance_ratio is not None:
        evidence.append(f"imbalance_ratio={profile.target.imbalance_ratio:.3f}")
    warnings: list[str] = []

    if n_unique <= 1:
        return ProblemDefinition(
            problem_type=None,
            confidence=Confidence.LOW,
            evidence=tuple(evidence + ["target has a single unique value"]),
            warnings=("Target is constant; no supervised problem can be defined.",),
            status=DecisionStatus.UNDECIDED,
        )

    if bool_like or n_unique == 2:
        return ProblemDefinition(
            problem_type=ProblemType.BINARY_CLASSIFICATION,
            confidence=Confidence.HIGH,
            evidence=tuple(evidence + ["exactly two unique target values"]),
            warnings=tuple(warnings),
            status=DecisionStatus.DECIDED,
        )

    integer_like = _looks_integer_like(target)
    # Integer dtype with values exactly 0..k-1 is the standard class-code encoding (iris, wine, digits).
    if numeric and 3 <= n_unique <= 10 and _is_class_code(target, n_unique):
        return ProblemDefinition(
            problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
            confidence=Confidence.MEDIUM,
            evidence=tuple(evidence + [f"integer class codes 0..{n_unique - 1}"]),
            warnings=(
                "Target looks like class codes 0..k-1, so it is treated as multiclass. "
                "If it is a quantity or an ordered score, pass problem_type='regression'.",
            ),
            status=DecisionStatus.DECIDED,
        )
    # Ambiguous: few integer-like values (e.g. a 1-5 rating) could be multiclass or ordinal regression.
    if numeric and integer_like and 3 <= n_unique <= 10:
        return ProblemDefinition(
            problem_type=None,
            confidence=Confidence.LOW,
            evidence=tuple(
                evidence
                + [
                    "integer-like target with low cardinality",
                    "could be multiclass classification or ordinal/regression",
                ]
            ),
            warnings=(
                "Ambiguous target. Pass problem_type='multiclass_classification' or "
                "problem_type='regression' to proceed.",
            ),
            status=DecisionStatus.UNDECIDED,
        )

    if (not numeric) and 3 <= n_unique <= 30:
        return ProblemDefinition(
            problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
            confidence=Confidence.HIGH if n_unique <= 15 else Confidence.MEDIUM,
            evidence=tuple(evidence + ["non-numeric target with moderate cardinality"]),
            warnings=tuple(warnings),
            status=DecisionStatus.DECIDED,
        )

    if numeric and n_unique > 20:
        return ProblemDefinition(
            problem_type=ProblemType.REGRESSION,
            confidence=Confidence.HIGH,
            evidence=tuple(evidence + ["numeric target with high cardinality"]),
            warnings=tuple(warnings),
            status=DecisionStatus.DECIDED,
        )

    if numeric and 11 <= n_unique <= 20 and integer_like:
        return ProblemDefinition(
            problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
            confidence=Confidence.MEDIUM,
            evidence=tuple(evidence + ["integer-like target with 11-20 classes; treated as multiclass"]),
            warnings=("If this target is numeric magnitude, pass problem_type='regression'.",),
            status=DecisionStatus.DECIDED,
        )

    if not numeric and n_unique > 30:
        return ProblemDefinition(
            problem_type=None,
            confidence=Confidence.LOW,
            evidence=tuple(evidence + ["very high-cardinality non-numeric target"]),
            warnings=("Target looks like an identifier or free text, not a label.",),
            status=DecisionStatus.UNDECIDED,
        )

    return ProblemDefinition(
        problem_type=None,
        confidence=Confidence.LOW,
        evidence=tuple(evidence + ["insufficient evidence to declare a problem type"]),
        warnings=("Pass problem_type explicitly.",),
        status=DecisionStatus.UNDECIDED,
    )
