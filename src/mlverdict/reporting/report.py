"""Markdown report covering the full Phase 1 decision path."""

from __future__ import annotations

from mlverdict.core.types import ModelDecisionRecord


def _bullets(items: tuple[str, ...] | list[str], empty: str = "- None.") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


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
    if record.final_test:
        exec_lines.append(
            f"Untouched final-test {record.final_test.metrics}."
        )
    if record.production:
        exec_lines.append(
            "Production readiness: **passed**." if record.production.passed else "Production readiness: **BLOCKED**."
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
                f"imbalance_ratio={dna.imbalance_ratio} missingness={dna.missingness}",
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
        _bullets(
            [
                f"metrics={record.final_test.metrics}",
                f"primary={record.final_test.primary_score:.4f}",
                f"validation_primary={record.final_test.validation_primary_score}",
                f"gap={record.final_test.gap}",
                f"n_rows={record.final_test.n_rows}",
            ]
            if record.final_test
            else ["Final test was not run."]
        ),
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
    ]
    return "\n".join(sections)
