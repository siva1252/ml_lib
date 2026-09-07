"""Plain-language metric help, test-set traps, and deploy next-steps."""

from __future__ import annotations

from typing import Any, Mapping

from mlverdict.core.enums import DecisionStatus, Severity
from mlverdict.core.types import FinalTestResult, LeakageReport, QualityReport

_WINDOWS_TRANS = str.maketrans(
    {
        "\u2192": "->",
        "\u2014": "-",
        "\u2013": "-",
        "\u2212": "-",
        "\u00d7": "x",
        "\u2022": "-",
        "\u00b7": "|",
        "\u2248": "~",
    }
)


def windows_safe(text: str) -> str:
    """Keep print(run) from crashing on Windows cp1252 consoles."""
    return text.translate(_WINDOWS_TRANS).encode("cp1252", errors="replace").decode("cp1252")


METRIC_PLAIN: dict[str, str] = {
    "pr_auc": "How well the model ranks the rare/positive class. Prefer this over accuracy when classes are imbalanced.",
    "roc_auc": "How well the model ranks positives above negatives across all cutoffs.",
    "accuracy": "Share of rows predicted correctly. Looks strong on imbalanced data even if the rare class is ignored.",
    "precision": "Of rows predicted positive, how many actually were (default 0.5 cutoff).",
    "recall": "Of actual positives, how many the model caught (default 0.5 cutoff).",
    "f1": "Balance of precision and recall at the default 0.5 cutoff.",
    "balanced_accuracy": "Average recall of each class. About 0.50 is a coin flip for binary problems.",
    "log_loss": "Quality of predicted probabilities. Lower is better.",
    "rmse": "Typical prediction error in the target's units. Lower is better.",
    "mae": "Average absolute error. Lower is better.",
    "r2": "Share of target variance explained. 1 is perfect; 0 is predicting the mean.",
    "mape": "Average percent error. Lower is better.",
}

_ZERO = 1e-12


def format_score(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number or number in {float("inf"), float("-inf")}:
        return "n/a"
    return f"{number:.{digits}f}"


def format_pct(value: float | None) -> str:
    if value is None or value != value:
        return "n/a"
    return f"{float(value):.1%}"


def format_latency(ms: float | None) -> str:
    if ms is None or ms != ms:
        return "n/a"
    return f"{float(ms):.2f} ms"


def format_imbalance(ratio: float | None) -> str:
    if ratio is None or ratio != ratio:
        return "balanced / not computed"
    return f"{float(ratio):.2f}:1 majority-to-minority"


def metric_help(name: str) -> str:
    return METRIC_PLAIN.get(name, "Score used to compare models on this problem.")


def _near_zero(value: float | None) -> bool:
    return value is not None and value == value and abs(float(value)) <= _ZERO


def interpret_final_test(
    metrics: Mapping[str, float] | None,
    *,
    is_classification: bool,
    n_rows: int | None = None,
) -> tuple[str, ...]:
    """Explain held-out numbers. Flags the accuracy-high / f1-zero trap."""
    if not metrics:
        return ()
    notes: list[str] = []
    if n_rows is not None and n_rows < 120:
        notes.append(
            f"Held-out test has {n_rows} rows - scores can jump around; treat them as a check, not a trophy."
        )
    if not is_classification:
        return tuple(notes)

    accuracy = metrics.get("accuracy")
    recall = metrics.get("recall")
    precision = metrics.get("precision")
    f1 = metrics.get("f1")
    balanced = metrics.get("balanced_accuracy")
    pr_auc = metrics.get("pr_auc")
    roc_auc = metrics.get("roc_auc")

    missed_positives = _near_zero(recall) and _near_zero(precision)
    if missed_positives or _near_zero(f1):
        notes.append(
            "At the default 0.5 cutoff the model never caught the positive class "
            "(precision = recall = f1 = 0). Ranking metrics can still look fine."
        )
        if accuracy is not None and accuracy >= 0.7:
            notes.append(
                f"Accuracy {format_pct(accuracy)} mostly reflects the majority class, "
                "not a useful alert rule."
            )
        if balanced is not None and abs(float(balanced) - 0.5) < 0.05:
            notes.append("balanced_accuracy ~ 0.50 is a coin flip on the class decision.")
        ranking_ok = (pr_auc is not None and pr_auc > 0.3) or (roc_auc is not None and roc_auc > 0.7)
        if ranking_ok:
            notes.append(
                "Probabilities still rank somewhat usefully. Do not deploy class alerts "
                "until you change the cutoff, rebalance, or add more signal."
            )
    return tuple(notes)


def interpret_gap(gap: float | None, n_rows: int | None = None) -> str | None:
    if gap is None or gap != gap:
        return None
    if gap > 0.08:
        return (
            "Cross-validation beat the held-out test. Possible overfitting, "
            "or a harder test split."
        )
    if gap < -0.08:
        extra = " With a small test set this is often noise, not proof the model improved."
        if n_rows is not None and n_rows < 120:
            return (
                "Held-out test beat cross-validation."
                + extra
            )
        return "Held-out test beat cross-validation. Confirm on more unseen data before celebrating."
    return "Held-out test is in the same ballpark as cross-validation."


def watchlist(quality: QualityReport | None, leakage: LeakageReport | None) -> tuple[str, ...]:
    rows: list[str] = []
    if quality is not None:
        for issue in quality.issues:
            if issue.severity in {Severity.HIGH_RISK, Severity.BLOCKER}:
                col = f"{issue.column}: " if issue.column else ""
                rows.append(f"{issue.severity.value}  {col}{issue.description}")
    if leakage is not None:
        for signal in leakage.signals:
            if signal.severity in {Severity.HIGH_RISK, Severity.BLOCKER}:
                col = f"{signal.column}: " if signal.column else ""
                rows.append(f"{signal.severity.value}  {col}{signal.description}")
    return tuple(rows)


def next_steps(
    *,
    status: DecisionStatus,
    has_artifact: bool,
    save_hint: str = "model.joblib",
) -> tuple[str, ...]:
    if status == DecisionStatus.UNDECIDED:
        return (
            "Pass problem_type='binary_classification', 'multiclass_classification', or 'regression'.",
            "Then run fit() again. There is nothing to save yet.",
        )
    if status == DecisionStatus.BLOCKED:
        return (
            "Fix the blockers in the report (data quality or production constraints).",
            "There is no deployable artifact from this run.",
        )
    if not has_artifact:
        return ("The run finished but no artifact was packaged. See limitations above.",)
    return (
        f'run.artifact().save("{save_hint}")',
        f'ModelArtifact.load("{save_hint}").predict(new_rows)',
        f'python -m mlverdict serve {save_hint}',
        "print(run.report())  # full written decision record",
    )


def metric_lines(metrics: Mapping[str, float], *, primary: str | None = None) -> tuple[str, ...]:
    lines: list[str] = []
    names = list(metrics.keys())
    if primary and primary in metrics:
        names = [primary] + [n for n in names if n != primary]
    for name in names:
        mark = "  (primary)" if primary and name == primary else ""
        help_text = metric_help(name)
        lines.append(f"{name:<22} {format_score(metrics[name])}{mark}")
        lines.append(f"  {help_text}")
    return tuple(lines)


def compact_metrics(metrics: Mapping[str, float] | None) -> str:
    if not metrics:
        return "none"
    parts = [f"{k}={format_score(v)}" for k, v in metrics.items()]
    return ", ".join(parts)


def as_mapping(metrics: Any) -> dict[str, float]:
    if not metrics:
        return {}
    return {str(k): float(v) for k, v in dict(metrics).items()}


def final_test_mapping(final_test: FinalTestResult | None) -> dict[str, float]:
    if final_test is None:
        return {}
    return as_mapping(final_test.metrics)
