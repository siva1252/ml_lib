"""Terminal verdict a user can actually read. print(run) uses this."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from mlverdict.core.enums import DecisionStatus
from mlverdict.reporting.insights import (
    compact_metrics,
    format_imbalance,
    format_latency,
    format_score,
    interpret_final_test,
    interpret_gap,
    metric_help,
    next_steps,
    watchlist,
    windows_safe,
)
from mlverdict.reporting.tables import ascii_table

if TYPE_CHECKING:
    from mlverdict.api.run import Run

_WIDTH = 76


def _rule(char: str = "=") -> str:
    return char * _WIDTH


def _title(text: str) -> str:
    return f"{_rule()}\n  {text}\n{_rule()}"


def _section(title: str) -> str:
    return f"\n{_rule('-')}\n  {title}\n{_rule('-')}"


def _wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(
        text,
        width=_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def _bullets(items: tuple[str, ...] | list[str], empty: str = "  (none)") -> str:
    if not items:
        return empty
    return "\n".join(_wrap(f"- {item}", indent="  ") for item in items)


def render_console(run: "Run") -> str:
    status = run.status
    decision = run.decision
    winner = decision.selected_model if decision else None
    metric = decision.primary_metric if decision else None
    val_score = decision.selected_primary_score if decision else None
    test = run.final_test
    dna = run.dna
    problem = run.problem

    banner_status = status.value if isinstance(status, DecisionStatus) else str(status)
    parts: list[str] = [
        _title(f"MLVerdict  |  {banner_status}"),
    ]

    if status == DecisionStatus.DECIDED and winner:
        parts.append("")
        parts.append(f"  Selected model : {winner}")
        if metric:
            parts.append(f"  Primary metric : {metric}  -  {metric_help(metric)}")
        parts.append(f"  Cross-validation: {format_score(val_score)}")
        if test is not None:
            parts.append(
                f"  Held-out test  : {format_score(test.primary_score)}   "
                f"({test.n_rows} untouched rows)"
            )
        prod = run.production
        if prod is not None:
            parts.append(
                "  Artifact       : ready to save"
                if prod.passed and run.artifact_obj is not None
                else "  Artifact       : not ready"
            )
    elif status == DecisionStatus.UNDECIDED:
        parts.append("")
        parts.append(_wrap("No model yet. The problem type is unclear - pass problem_type and fit again."))
        if run.notes:
            parts.append(_bullets(list(run.notes)))
        elif decision and decision.reasons:
            parts.append(_bullets(list(decision.reasons)))
    else:
        parts.append("")
        parts.append(_wrap("Run is blocked. Nothing was packaged for deployment."))
        if decision and decision.reasons:
            parts.append(_bullets(list(decision.reasons)))
        elif run.notes:
            parts.append(_bullets(list(run.notes)))

    parts.append(_section("Dataset"))
    if dna is not None:
        problem_name = (
            problem.problem_type.value.replace("_", " ")
            if problem and problem.problem_type
            else "undecided"
        )
        parts.append(_wrap(f"Target     : {dna.target_name}  ({dna.target_kind}, {dna.target_cardinality} unique values)"))
        parts.append(_wrap(f"Problem    : {problem_name}"))
        parts.append(
            _wrap(
                f"Size       : {dna.n_rows} modeling rows x {dna.n_features} features  "
                f"(scale={dna.scale.value})"
            )
        )
        parts.append(_wrap(f"Imbalance  : {format_imbalance(dna.imbalance_ratio)}"))
        parts.append(_wrap(f"Missing    : {dna.missingness}"))
        if dna.potential_id_columns:
            parts.append(_wrap(f"IDs seen   : {', '.join(dna.potential_id_columns)} (dropped from features)"))
    else:
        parts.append("  No dataset profile (run stopped early).")

    if run.metric_plan is not None:
        parts.append(_section("Why this metric"))
        primary = run.metric_plan.primary.name
        parts.append(_wrap(f"Primary: {primary}"))
        parts.append(_wrap(metric_help(primary)))
        if run.metric_plan.evidence:
            parts.append(_bullets(list(run.metric_plan.evidence)))

    parts.append(_section("Leaderboard"))
    parts.append(_wrap("Ranked by evidence (performance + stability + generalization + latency + complexity)."))
    parts.append(_wrap("Composite is a ranking aid, not a published accuracy."))
    parts.append("")
    parts.append(_leaderboard_block(run))

    if decision and decision.reasons and status == DecisionStatus.DECIDED:
        parts.append(_section("Why this model"))
        parts.append(_bullets(list(decision.reasons)))
        if decision.tradeoffs:
            parts.append("")
            parts.append("  Tradeoffs")
            parts.append(_bullets(list(decision.tradeoffs)))

    parts.append(_section("Held-out test (scored once, after selection)"))
    if test is None:
        parts.append("  Not run.")
    else:
        parts.append(_wrap(f"Rows: {test.n_rows}"))
        parts.append(
            _wrap(
                f"Primary {metric or 'score'}: CV {format_score(test.validation_primary_score)}  ->  "
                f"test {format_score(test.primary_score)}"
            )
        )
        gap_note = interpret_gap(test.gap, test.n_rows)
        if gap_note:
            parts.append(_wrap(gap_note))
        parts.append("")
        parts.append(_metrics_table(test.metrics, primary=metric))
        is_cls = bool(problem and problem.is_classification)
        traps = interpret_final_test(test.metrics, is_classification=is_cls, n_rows=test.n_rows)
        if traps:
            parts.append("")
            parts.append("  ATTENTION")
            parts.append(_bullets(list(traps)))

    flags = watchlist(run.quality, run.leakage)
    parts.append(_section("Watchlist"))
    if flags:
        parts.append(_bullets(list(flags)))
        parts.append("")
        parts.append(_wrap("These are warnings, not a leakage-free certificate. Identifiers are dropped automatically."))
    else:
        parts.append("  No high-risk quality or leakage flags.")

    parts.append(_section("Production readiness"))
    prod = run.production
    if prod is None:
        parts.append("  Not evaluated.")
    else:
        n_ok = sum(1 for c in prod.checks if c.passed)
        n = len(prod.checks)
        parts.append(_wrap(f"{n_ok}/{n} checks passed. This is packaging/plumbing, not business quality."))
        failed = [c for c in prod.checks if not c.passed]
        if failed:
            parts.append(_bullets([f"{c.name}: {c.detail}" for c in failed]))
        else:
            parts.append("  Pipeline saves, reloads, and predicts. Ready to export.")

    parts.append(_section("What to do next"))
    parts.append(_bullets(list(next_steps(status=status, has_artifact=run.artifact_obj is not None))))
    parts.append("")
    parts.append(_rule())
    return windows_safe("\n".join(parts).rstrip() + "\n")


def _hpo_names(run: "Run") -> set[str]:
    return {e.model_name for e in run.experiments if e.optimized}


def _leaderboard_block(run: "Run") -> str:
    rows = []
    tuned = _hpo_names(run)
    ranked = sorted(
        run.evaluations,
        key=lambda e: (e.passes_constraints, e.composite_score),
        reverse=True,
    )
    if not ranked:
        if run.experiments:
            fallback = []
            for exp in run.experiments:
                fallback.append(
                    [
                        "-",
                        exp.model_name,
                        format_score(exp.mean_score),
                        format_score(exp.std_score),
                        format_latency(exp.infer_latency_ms),
                        exp.status.value,
                        "HPO" if exp.optimized else "baseline",
                    ]
                )
            return ascii_table(
                ("#", "Model", "CV score", "Std", "Latency", "Status", "Stage"),
                fallback,
                aligns=("r", "l", "r", "r", "r", "l", "l"),
            )
        return "  No models were evaluated."

    for i, ev in enumerate(ranked, start=1):
        rows.append(
            [
                str(i),
                ev.model_name,
                format_score(ev.primary_score),
                format_score(ev.stability_score, 3),
                format_latency(ev.latency_ms),
                format_score(ev.composite_score, 3),
                "yes" if ev.passes_constraints else "NO",
                "HPO" if ev.model_name in tuned else "base",
            ]
        )
    return ascii_table(
        ("#", "Model", "CV score", "Stable", "Latency", "Evidence", "OK?", "Stage"),
        rows,
        aligns=("r", "l", "r", "r", "r", "r", "c", "l"),
    )


def _metrics_table(metrics: dict[str, float], *, primary: str | None) -> str:
    if not metrics:
        return "  no metrics"
    names = list(metrics.keys())
    if primary and primary in metrics:
        names = [primary] + [n for n in names if n != primary]
    rows = []
    for name in names:
        role = "primary" if primary and name == primary else "check"
        rows.append([name, format_score(metrics[name]), role])
    return ascii_table(("Metric", "Value", "Role"), rows, aligns=("l", "r", "l"))


def render_summary(run: "Run") -> str:
    """One short block for logs."""
    best = run.best()
    if not best:
        return f"MLVerdict {run.status.value}: no model selected."
    test = best.get("final_test_score")
    extra = ""
    if run.final_test and run.final_test.metrics:
        extra = f" | test {compact_metrics(run.final_test.metrics)}"
    return windows_safe(
        f"MLVerdict {best['status']}: {best['model']}  "
        f"{best['primary_metric']}  CV={format_score(best['validation_score'])}  "
        f"test={format_score(test)}{extra}"
    )
