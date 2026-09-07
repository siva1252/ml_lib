"""Markdown report covering the full Phase 1 decision path."""

from __future__ import annotations

from mlverdict.core.enums import DecisionStatus
from mlverdict.core.types import ModelDecisionRecord
from mlverdict.reporting.insights import (
    format_imbalance,
    format_score,
    interpret_final_test,
    interpret_gap,
    metric_help,
    next_steps,
    watchlist,
    windows_safe,
)


def _bullets(items: tuple[str, ...] | list[str], empty: str = "- None.") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def _metric_md(metrics: dict[str, float] | None, *, primary: str | None) -> list[str]:
    if not metrics:
        return ["No metrics recorded."]
    names = list(metrics.keys())
    if primary and primary in metrics:
        names = [primary] + [n for n in names if n != primary]
    lines: list[str] = []
    for name in names:
        tag = " **(primary)**" if primary and name == primary else ""
        lines.append(f"**{name}**{tag}: {format_score(metrics[name])} - {metric_help(name)}")
    return lines


def render_report(record: ModelDecisionRecord) -> str:
    d = record.decision
    dna = record.dataset_dna
    problem = record.problem
    summary_status = d.status.value
    selected = d.selected_model or "No model selected"
    exec_lines = [
        f"MLVerdict status: **{summary_status}**.",
        f"Selected model: **{selected}**.",
    ]
    if d.primary_metric and d.selected_primary_score is not None:
        exec_lines.append(
            f"Primary metric **{d.primary_metric}** validation score: **{d.selected_primary_score:.4f}**."
        )
        exec_lines.append(metric_help(d.primary_metric))
    if record.final_test:
        exec_lines.append(
            f"Held-out test **{d.primary_metric or 'primary'}**: "
            f"**{format_score(record.final_test.primary_score)}** "
            f"({record.final_test.n_rows} untouched rows)."
        )
    if record.production:
        exec_lines.append(
            "Production packaging: **passed**." if record.production.passed else "Production packaging: **BLOCKED**."
        )

    quality_lines = [
        f"{i.severity.value} [{i.issue_type}] {i.column or '-'}: {i.description} ({i.evidence})"
        for i in record.quality.issues
    ]
    leak_lines = [
        f"{s.severity.value} [{s.signal_type}] {s.column or '-'}: {s.description}"
        for s in record.leakage.signals
    ]
    candidate_lines = [f"Included: {n}" for n in record.candidates] + [
        f"Excluded: {n}" for n in record.excluded_candidates
    ]
    exp_lines = [
        f"{e.model_name} ({'HPO' if e.optimized else 'baseline'}): "
        f"mean={e.mean_score:.4f} std={e.std_score:.4f} latency={e.infer_latency_ms:.2f}ms status={e.status.value}"
        + (f" error={e.error}" if e.error else "")
        for e in record.experiments
    ]
    ev_lines = [
        f"{e.model_name}: primary={e.primary_score:.4f} stability={e.stability_score:.3f} "
        f"gap_score={e.generalization_score:.3f} latency={e.latency_ms:.2f}ms "
        f"constraints={'pass' if e.passes_constraints else 'FAIL'} composite={e.composite_score:.4f}"
        for e in record.evaluations
    ]
    rejected = [f"{r.model_name}: {r.reason}" for r in d.rejected]

    test_lines: list[str]
    attention: list[str] = []
    if record.final_test:
        ft = record.final_test
        test_lines = [
            f"untouched rows: {ft.n_rows}",
            f"primary test score: {format_score(ft.primary_score)}",
            f"primary validation score: {format_score(ft.validation_primary_score)}",
            f"gap (validation - test): {format_score(ft.gap)}",
        ]
        gap_note = interpret_gap(ft.gap, ft.n_rows)
        if gap_note:
            test_lines.append(gap_note)
        test_lines.extend(_metric_md(ft.metrics, primary=d.primary_metric))
        attention = list(
            interpret_final_test(
                ft.metrics,
                is_classification=problem.is_classification,
                n_rows=ft.n_rows,
            )
        )
    else:
        test_lines = ["Final test was not run."]

    flags = watchlist(record.quality, record.leakage)
    deploy = next_steps(
        status=d.status if isinstance(d.status, DecisionStatus) else DecisionStatus(d.status),
        has_artifact=record.production is not None and record.production.passed and d.status == DecisionStatus.DECIDED,
    )

    sections = [
        "# MLVerdict Report",
        "",
        "## Executive summary",
        _bullets(exec_lines),
        "",
        "## Dataset DNA",
        _bullets(
            [
                f"rows={dna.n_rows}, features={dna.n_features}, scale={dna.scale.value}",
                f"target={dna.target_name} kind={dna.target_kind} cardinality={dna.target_cardinality}",
                f"imbalance={format_imbalance(dna.imbalance_ratio)} missingness={dna.missingness}",
                f"iid={dna.iid_assumption.value} entity_hint={dna.entity_hint} time_hint={dna.time_hint}",
                f"potential_ids={list(dna.potential_id_columns)}",
                f"high_cardinality={list(dna.high_cardinality_columns)}",
            ]
        ),
        "",
        "## Problem",
        _bullets(
            [
                f"type={problem.problem_type.value if problem.problem_type else None}",
                f"confidence={problem.confidence.value} status={problem.status.value} override={problem.user_override}",
                *problem.evidence,
                *problem.warnings,
            ]
        ),
        "",
        "## Data quality",
        _bullets(quality_lines),
        "",
        "## Leakage signals",
        f"_{record.leakage.claim}_",
        "",
        _bullets(leak_lines),
        "",
        "## Validation",
        _bullets(
            [
                f"strategy={record.validation.strategy.value} splits={record.validation.n_splits}",
                f"group_col={record.validation.group_column} time_col={record.validation.time_column}",
                *record.validation.evidence,
                *record.validation.warnings,
            ]
        ),
        "",
        "## Metrics",
        _bullets(
            [
                f"primary={record.metrics.primary.name} (user_override={record.metrics.user_override})",
                metric_help(record.metrics.primary.name),
                f"secondary={[m.name for m in record.metrics.secondary]}",
                *record.metrics.evidence,
            ]
        ),
        "",
        "## Candidates",
        _bullets(candidate_lines),
        "",
        "## Experiments and HPO",
        _bullets(exp_lines),
        "",
        "## Evaluation (performance + stability + constraints)",
        _bullets(ev_lines),
        "",
        "## Selected model",
        _bullets(d.reasons),
        "",
        "## Rejected alternatives",
        _bullets(rejected),
        "",
        "## Tradeoffs",
        _bullets(d.tradeoffs),
        "",
        "## Final untouched test",
        _bullets(test_lines),
        "",
        "## Attention",
        _bullets(attention, empty="- No threshold/majority-class traps flagged."),
        "",
        "## Watchlist (high-risk only)",
        _bullets(list(flags), empty="- No high-risk quality or leakage flags."),
        "",
        "## Assumptions",
        _bullets(d.assumptions),
        "",
        "## Limitations",
        _bullets(record.limitations),
        "",
        "## Production readiness",
        _bullets(
            [f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.detail}" for c in record.production.checks]
            if record.production
            else ["Not evaluated."]
        ),
        "",
        "## What to do next",
        _bullets(list(deploy)),
        "",
    ]
    return windows_safe("\n".join(sections))
