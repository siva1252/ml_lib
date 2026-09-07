"""Resolve whether fit() is supervised Phase 1 or an unsupervised workflow."""

from __future__ import annotations

from dataclasses import dataclass

from mlverdict.core.enums import DecisionStatus, UnsupervisedTask
from mlverdict.core.exceptions import ConfigurationError

_TASK_VALUES = tuple(item.value for item in UnsupervisedTask)


@dataclass(frozen=True)
class FitIntent:
    """Entry contract. halt_status set means Phase 1 supervised training must not run."""

    supervised: bool
    target: str | None
    unsupervised_task: UnsupervisedTask | None
    halt_status: DecisionStatus | None
    notes: tuple[str, ...]
    learning_mode: str


def normalize_target(target: str | None) -> str | None:
    if target is None:
        return None
    name = str(target).strip()
    return name or None


def parse_unsupervised_task(task: str | UnsupervisedTask | None) -> UnsupervisedTask | None:
    if task is None:
        return None
    if isinstance(task, UnsupervisedTask):
        return task
    key = str(task).strip().lower()
    if not key:
        return None
    for item in UnsupervisedTask:
        if item.value == key:
            return item
    raise ConfigurationError(
        f"Unknown task '{task}'. Supported unsupervised tasks: {', '.join(_TASK_VALUES)}."
    )


def resolve_fit_intent(
    target: str | None,
    task: str | UnsupervisedTask | None = None,
) -> FitIntent:
    target_name = normalize_target(target)
    parsed_task = parse_unsupervised_task(task)

    if target_name is not None and parsed_task is not None:
        raise ConfigurationError(
            "Pass a target for supervised learning, or task= for an unsupervised "
            "objective, not both."
        )

    if target_name is not None:
        return FitIntent(
            supervised=True,
            target=target_name,
            unsupervised_task=None,
            halt_status=None,
            notes=(),
            learning_mode="supervised",
        )

    if parsed_task is None:
        return FitIntent(
            supervised=False,
            target=None,
            unsupervised_task=None,
            halt_status=DecisionStatus.UNDECIDED,
            notes=(
                "No target column was provided, so this is not a supervised workflow.",
                "A missing target does not select clustering, anomaly detection, or "
                "dimensionality reduction.",
                "Pass target='<column>' for supervised Phase 1, or set task to "
                "'clustering', 'anomaly_detection', or 'dimensionality_reduction'.",
            ),
            learning_mode="unsupervised_candidate",
        )

    return FitIntent(
        supervised=False,
        target=None,
        unsupervised_task=parsed_task,
        halt_status=None,
        notes=(),
        learning_mode="unsupervised",
    )
