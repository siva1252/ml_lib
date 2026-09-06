"""Detect the ML problem from target evidence. Never silently force a low-confidence call."""

from __future__ import annotations

import pandas as pd

from mlverdict.core.enums import Confidence, DecisionStatus, ProblemType
from mlverdict.core.exceptions import ConfigurationError
from mlverdict.core.types import DatasetProfile, ProblemDefinition

_VALID_OVERRIDES = {item.value: item for item in ProblemType}


def _looks_integer_like(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    clean = series.dropna()
    if clean.empty:
        return False
    return bool(((clean * 10).round() / 10 == clean.round()).mean() > 0.98)


def detect_problem(
    profile: DatasetProfile,
    target: pd.Series,
    *,
    user_problem_type: str | ProblemType | None = None,
) -> ProblemDefinition:
    if user_problem_type is not None:
        key = user_problem_type.value if isinstance(user_problem_type, ProblemType) else str(user_problem_type)
        if key not in _VALID_OVERRIDES:
            raise ConfigurationError(f"Unknown problem_type '{key}'.")
        return ProblemDefinition(
            problem_type=_VALID_OVERRIDES[key],
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
    # Ambiguous: few integer codes could be multiclass or ordinal regression.
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
