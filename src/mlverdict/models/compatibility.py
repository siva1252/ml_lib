"""Compatibility filters: installed extras, scale, constraints, problem type."""

from __future__ import annotations

from mlverdict.core.config import Constraints, VerdictConfig
from mlverdict.core.enums import DatasetScale
from mlverdict.core.types import DatasetDNA, ProblemDefinition
from mlverdict.models.registry import is_available


def compatible(
    entry: dict,
    dna: DatasetDNA,
    problem: ProblemDefinition,
    constraints: Constraints | None,
    config: VerdictConfig | None = None,
) -> tuple[bool, str]:
    config = config or VerdictConfig()
    if problem.is_unsupervised:
        task = problem.problem_type.value if problem.problem_type else "unsupervised"
    else:
        task = "regression" if problem.is_regression else "classification"
    if task not in entry["tasks"]:
        return False, f"not applicable to {task}"
    if not is_available(entry["key"]):
        return False, "library not installed"
    if constraints and constraints.require_explainable and not entry["explainable"]:
        return False, "require_explainable=True"
    if constraints and constraints.max_complexity == "low" and entry["complexity"] != "low":
        return False, "max_complexity=low"
    if entry["key"] in {"svm", "one_class_svm"} and dna.n_rows > config.max_rows_for_svm:
        return False, f"SVM training is O(n^2); {dna.n_rows} rows > max_rows_for_svm={config.max_rows_for_svm}"
    if entry["key"] == "knn" and dna.n_rows > config.max_rows_for_knn:
        return False, f"KNN inference scans the training set; {dna.n_rows} rows > max_rows_for_knn={config.max_rows_for_knn}"
    if dna.scale == DatasetScale.TINY and entry["family"].value == "boosting":
        return False, "boosting is excessive on tiny datasets"
    if constraints and constraints.max_latency_ms is not None and constraints.max_latency_ms < 5:
        if entry["complexity"] == "high":
            return False, "latency budget is too tight for high-complexity models"
    return True, "compatible"
