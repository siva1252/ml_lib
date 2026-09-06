"""Choose a validation strategy from DNA + problem + user columns. Hints are not silent facts."""

from __future__ import annotations

from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import DatasetScale, IIDAssumption, ValidationStrategy
from mlverdict.core.types import DatasetDNA, ProblemDefinition, ValidationPlan


def select_validation_plan(
    dna: DatasetDNA,
    problem: ProblemDefinition,
    config: VerdictConfig | None = None,
    *,
    group_col: str | None = None,
    time_col: str | None = None,
    user_strategy: str | None = None,
) -> ValidationPlan:
    config = config or VerdictConfig()
    n_splits = config.cv_splits
    if dna.scale == DatasetScale.TINY:
        n_splits = min(n_splits, 3)
    if dna.n_rows < config.min_rows_for_cv:
        n_splits = 2

    warnings: list[str] = []
    evidence: list[str] = [
        f"scale={dna.scale.value}",
        f"iid={dna.iid_assumption.value}",
        f"imbalance_ratio={dna.imbalance_ratio}",
        f"problem={problem.problem_type.value if problem.problem_type else None}",
    ]

    if user_strategy:
        strategy = ValidationStrategy(user_strategy)
        return ValidationPlan(
            strategy=strategy,
            n_splits=n_splits,
            test_size=config.test_size,
            group_column=group_col,
            time_column=time_col,
            stratify=strategy == ValidationStrategy.STRATIFIED_KFOLD,
            evidence=tuple(evidence + [f"user override strategy={user_strategy}"]),
            warnings=("Validation strategy was set by the user.",),
            shuffle=strategy != ValidationStrategy.TIME_BASED,
        )

    if time_col:
        return ValidationPlan(
            strategy=ValidationStrategy.TIME_BASED,
            n_splits=max(2, min(n_splits, 4)),
            test_size=config.test_size,
            group_column=group_col,
            time_column=time_col,
            stratify=False,
            evidence=tuple(evidence + [f"confirmed time column '{time_col}'"]),
            warnings=tuple(warnings),
            shuffle=False,
        )

    if group_col:
        return ValidationPlan(
            strategy=ValidationStrategy.GROUP_KFOLD,
            n_splits=n_splits,
            test_size=config.test_size,
            group_column=group_col,
            time_column=time_col,
            stratify=False,
            evidence=tuple(evidence + [f"confirmed group column '{group_col}'"]),
            warnings=tuple(warnings),
            shuffle=True,
        )

    if dna.entity_hint and dna.iid_assumption != IIDAssumption.LIKELY:
        warnings.append(
            f"Repeated-entity hint '{dna.entity_hint}' was not confirmed. "
            "Using a non-grouped split. Pass group_col to enable grouped validation."
        )

    if dna.time_hint and not time_col:
        warnings.append(
            f"Datetime hint '{dna.time_hint}' was not confirmed as the event time. "
            "Pass time_col to enable time-based validation."
        )

    imbalanced = bool(dna.imbalance_ratio and dna.imbalance_ratio >= 3.0)
    if problem.is_classification and dna.n_rows >= config.min_rows_for_cv:
        return ValidationPlan(
            strategy=ValidationStrategy.STRATIFIED_KFOLD,
            n_splits=n_splits,
            test_size=config.test_size,
            group_column=None,
            time_column=None,
            stratify=True,
            evidence=tuple(
                evidence
                + [
                    "classification + no confirmed group/time",
                    f"imbalanced={imbalanced}",
                ]
            ),
            warnings=tuple(warnings),
            shuffle=True,
        )

    if dna.n_rows < config.min_rows_for_cv:
        return ValidationPlan(
            strategy=ValidationStrategy.HOLDOUT,
            n_splits=1,
            test_size=max(config.test_size, 0.25),
            group_column=None,
            time_column=None,
            stratify=problem.is_classification,
            evidence=tuple(evidence + ["too few rows for k-fold"]),
            warnings=tuple(warnings),
            shuffle=True,
        )

    return ValidationPlan(
        strategy=ValidationStrategy.KFOLD,
        n_splits=n_splits,
        test_size=config.test_size,
        group_column=None,
        time_column=None,
        stratify=False,
        evidence=tuple(evidence + ["regression or IID fallback"]),
        warnings=tuple(warnings),
        shuffle=True,
    )
