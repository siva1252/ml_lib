"""Unsupervised lifecycle: same gates as supervised, without a labeled target."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from mlverdict.api.run import Run
from mlverdict.core.config import Constraints, VerdictConfig
from mlverdict.core.enums import (
    Confidence,
    DatasetScale,
    DecisionStatus,
    IIDAssumption,
    ProblemType,
    Severity,
    UnsupervisedTask,
)
from mlverdict.core.types import (
    SCHEMA_VERSION,
    DatasetDNA,
    FeatureSchema,
    FinalTestResult,
    ModelDecision,
    ProblemDefinition,
    ValidationPlan,
)
from mlverdict.data.dna import build_dna
from mlverdict.data.profiler import profile_dataset
from mlverdict.decision.record import build_decision_record
from mlverdict.decision.selector import select_model
from mlverdict.evaluation.evaluator import evaluate_experiments
from mlverdict.experiments.scoring import extract_score
from mlverdict.experiments.unsupervised import compute_unsupervised_metrics, run_unsupervised_experiments
from mlverdict.metrics.intelligence import select_metric_plan
from mlverdict.models.candidates import select_candidates
from mlverdict.preprocessing.pipeline import build_pipeline
from mlverdict.preprocessing.planner import plan_preprocessing
from mlverdict.problem.detector import unsupervised_problem
from mlverdict.problem.intent import FitIntent
from mlverdict.production.artifact import ModelArtifact
from mlverdict.production.readiness import assess_readiness
from mlverdict.quality.analyzer import analyze_quality
from mlverdict.quality.leakage import detect_leakage
from mlverdict.reporting.report import render_report
from mlverdict.validation.splitter import isolate_final_test
from mlverdict.validation.strategy import select_validation_plan


def fit_unsupervised(
    *,
    frame: pd.DataFrame,
    intent: FitIntent,
    config: VerdictConfig,
    constraints: Constraints,
    enable_hpo: bool,
    group_col: str | None = None,
    time_col: str | None = None,
    primary_metric: str | None = None,
    validation_strategy: str | None = None,
    feature_columns: list[str] | None = None,
) -> Run:
    assert intent.unsupervised_task is not None
    extras = {
        "learning_mode": "unsupervised",
        "unsupervised_task": intent.unsupervised_task.value,
    }
    provisional = _provisional_plan(frame, config, group_col, time_col, validation_strategy)
    split = isolate_final_test(frame, None, provisional, config.random_state)
    train = split.train
    test = split.test

    profile = profile_dataset(train, None, config)
    dna = build_dna(profile, config, group_col=group_col, time_col=time_col)
    problem = unsupervised_problem(intent.unsupervised_task)
    quality = analyze_quality(train, profile, config)
    leakage = detect_leakage(train, profile, dna, None, group_col=group_col)

    if quality.blockers:
        decision = ModelDecision(
            status=DecisionStatus.BLOCKED,
            selected_model=None,
            reasons=tuple(f"BLOCKER: {b.description}" for b in quality.blockers),
            tradeoffs=(),
            rejected=(),
            assumptions=(),
            limitations=("Quality blockers must be resolved before modeling.",),
        )
        return _early(
            DecisionStatus.BLOCKED,
            profile,
            dna,
            problem,
            quality,
            leakage,
            decision=decision,
            notes=tuple(b.description for b in quality.blockers),
            extras=extras,
        )

    validation = select_validation_plan(
        dna,
        problem,
        config,
        group_col=group_col,
        time_col=time_col,
        user_strategy=validation_strategy,
    )
    metric_plan = select_metric_plan(dna, problem, user_metric=primary_metric)
    candidates = select_candidates(dna, problem, metric_plan, constraints)

    extra_drop = tuple(c for c in (group_col,) if c)
    feature_frame = _feature_frame(train, feature_columns, extra_drop)
    groups = train[group_col] if group_col and group_col in train.columns else None

    experiments = run_unsupervised_experiments(
        feature_frame,
        candidates.included,
        profile,
        dna,
        problem,
        metric_plan,
        validation,
        config,
        groups=groups,
        extra_drop=extra_drop,
        enable_hpo=enable_hpo,
    )
    evaluations = evaluate_experiments(experiments, config, constraints)
    assumptions = _assumptions(dna, validation, group_col, time_col, intent.unsupervised_task)
    limitations = _limitations(quality, leakage, validation)
    decision = select_model(evaluations, metric_plan, assumptions=assumptions, limitations=limitations)

    final_test = None
    artifact = None
    production = None
    if decision.status == DecisionStatus.DECIDED and decision.selected_model:
        winner = next(c for c in candidates.included if c.name == decision.selected_model)
        winner_exp = next(e for e in experiments if e.model_name == winner.name)
        pre_plan = plan_preprocessing(profile, dna, winner.family, config, extra_drop=extra_drop)
        pipeline = build_pipeline(
            pre_plan,
            winner,
            problem.problem_type,
            config.random_state,
            winner_exp.configuration,
        )
        pipeline.fit(feature_frame)
        if len(test):
            X_test = _feature_frame(test, feature_columns, extra_drop)
            test_metrics = compute_unsupervised_metrics(pipeline, X_test, problem)
            primary = extract_score(test_metrics, metric_plan)
            raw_primary = test_metrics.get(metric_plan.primary.name, primary)
            gap = None
            if decision.selected_primary_score is not None and raw_primary == raw_primary:
                gap = float(decision.selected_primary_score - primary)
            final_test = FinalTestResult(
                metrics=test_metrics,
                primary_score=float(raw_primary) if raw_primary == raw_primary else float("nan"),
                validation_primary_score=decision.selected_primary_score,
                gap=gap,
                n_rows=len(test),
            )
        schema = FeatureSchema(
            feature_names=tuple(feature_frame.columns),
            numeric_columns=pre_plan.numeric_columns,
            categorical_columns=pre_plan.categorical_columns,
            datetime_columns=pre_plan.datetime_columns,
            target_name="",
            problem_type=problem.problem_type.value,
            dtypes=tuple((c, str(feature_frame[c].dtype)) for c in feature_frame.columns),
        )
        artifact = ModelArtifact(
            pipeline=pipeline,
            schema=schema,
            metadata={
                "model": winner.name,
                "infer_latency_ms": winner_exp.infer_latency_ms,
                "schema_version": SCHEMA_VERSION,
                "learning_mode": "unsupervised",
                "unsupervised_task": intent.unsupervised_task.value,
            },
            metrics=final_test.metrics if final_test else winner_exp.metrics,
            decision_record={},
            configuration=asdict(config),
        )
        sample = feature_frame.head(min(8, len(feature_frame)))
        production = assess_readiness(artifact, sample, schema)
        if production.blocked:
            decision = ModelDecision(
                status=DecisionStatus.BLOCKED,
                selected_model=decision.selected_model,
                reasons=decision.reasons + ("Production readiness blocked deployment.",),
                tradeoffs=decision.tradeoffs,
                rejected=decision.rejected,
                assumptions=decision.assumptions,
                limitations=decision.limitations + ("Artifact failed the readiness gate.",),
                primary_metric=decision.primary_metric,
                selected_primary_score=decision.selected_primary_score,
            )

    record = build_decision_record(
        dna=dna,
        problem=problem,
        quality=quality,
        leakage=leakage,
        validation=validation,
        metrics=metric_plan,
        candidates=candidates,
        experiments=experiments,
        evaluations=evaluations,
        decision=decision,
        final_test=final_test,
        constraints=asdict(constraints),
        production=production,
    )
    if artifact is not None:
        artifact.decision_record = record
    report_text = render_report(record)
    status = decision.status
    return Run(
        status=status,
        profile=profile,
        dna=dna,
        problem=problem,
        quality=quality,
        leakage=leakage,
        validation=validation,
        metric_plan=metric_plan,
        candidates=candidates,
        experiments=experiments,
        evaluations=evaluations,
        decision=decision,
        decision_record=record,
        final_test=final_test,
        production=production,
        artifact_obj=artifact if status == DecisionStatus.DECIDED else artifact,
        report_text=report_text,
        extras=extras,
    )


def _feature_frame(
    frame: pd.DataFrame,
    feature_columns: list[str] | None,
    extra_drop: tuple[str, ...],
) -> pd.DataFrame:
    drop = set(extra_drop)
    cols = list(feature_columns) if feature_columns else [c for c in frame.columns if c not in drop]
    cols = [c for c in cols if c not in drop and c in frame.columns]
    return frame.loc[:, cols].copy()


def _provisional_plan(
    frame: pd.DataFrame,
    config: VerdictConfig,
    group_col: str | None,
    time_col: str | None,
    user_strategy: str | None,
) -> ValidationPlan:
    dummy_problem = ProblemDefinition(
        problem_type=ProblemType.CLUSTERING,
        confidence=Confidence.MEDIUM,
        evidence=("provisional unsupervised split only — not a modeling decision",),
        warnings=(),
        status=DecisionStatus.DECIDED,
    )
    dummy_dna = DatasetDNA(
        schema_version=SCHEMA_VERSION,
        n_rows=len(frame),
        n_features=frame.shape[1],
        target_name="",
        target_kind="none",
        target_cardinality=0,
        imbalance_ratio=None,
        missingness="none",
        has_duplicates=False,
        has_datetime=time_col is not None,
        has_potential_ids=False,
        potential_id_columns=(),
        high_cardinality_columns=(),
        constant_columns=(),
        near_constant_columns=(),
        entity_hint=group_col,
        time_hint=time_col,
        iid_assumption=IIDAssumption.UNLIKELY if (group_col or time_col) else IIDAssumption.LIKELY,
        scale=DatasetScale.SMALL if len(frame) < 2000 else DatasetScale.MEDIUM,
        numeric_fraction=0.5,
        categorical_fraction=0.5,
    )
    return select_validation_plan(
        dummy_dna,
        dummy_problem,
        config,
        group_col=group_col,
        time_col=time_col,
        user_strategy=user_strategy,
    )


def _assumptions(dna, validation, group_col, time_col, task: UnsupervisedTask) -> tuple[str, ...]:
    items = [
        f"Unsupervised task is {task.value}; there is no labeled target.",
        f"IID assumption treated as {dna.iid_assumption.value}.",
        f"Validation strategy is {validation.strategy.value}.",
        "Internal scores (silhouette, reconstruction, score spread) are not labeled accuracy.",
    ]
    if dna.entity_hint and not group_col:
        items.append(f"Entity hint '{dna.entity_hint}' was not confirmed as group_col.")
    if dna.time_hint and not time_col:
        items.append(f"Time hint '{dna.time_hint}' was not confirmed as time_col.")
    return tuple(items)


def _limitations(quality, leakage, validation) -> tuple[str, ...]:
    items = [
        "Final test was isolated before modeling decisions and was scored once after selection.",
        "Leakage analysis reports signals, not a leakage-free certificate.",
        "Unsupervised metrics do not prove a business outcome without labels.",
    ]
    if quality.high_risks:
        items.append(f"{len(quality.high_risks)} high-risk quality issue(s) remain in the decision record.")
    if validation.warnings:
        items += list(validation.warnings)
    high_leak = [s for s in leakage.signals if s.severity in {Severity.HIGH_RISK, Severity.BLOCKER}]
    if high_leak:
        items.append(f"{len(high_leak)} high-risk leakage signal(s) should be reviewed before deployment.")
    return tuple(items)


def _early(
    status: DecisionStatus,
    profile,
    dna,
    problem,
    quality,
    leakage,
    *,
    decision: ModelDecision | None = None,
    notes: tuple[str, ...] = (),
    extras: dict[str, Any] | None = None,
) -> Run:
    if decision is None:
        decision = ModelDecision(
            status=status,
            selected_model=None,
            reasons=notes,
            tradeoffs=(),
            rejected=(),
            assumptions=(),
            limitations=notes,
        )
    return Run(
        status=status,
        profile=profile,
        dna=dna,
        problem=problem,
        quality=quality,
        leakage=leakage,
        decision=decision,
        notes=notes,
        extras=extras or {},
        report_text="MLVerdict " + status.value + "\n" + "\n".join(notes),
    )
