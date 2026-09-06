"""Public orchestrator. Complexity stays behind fit() / best() / verdict() / report() / artifact()."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from mlverdict.api.run import Run
from mlverdict.core.config import Constraints, VerdictConfig
from mlverdict.core.enums import DecisionStatus, ProblemType, Severity
from mlverdict.core.reproducibility import seed_everything
from mlverdict.core.types import (
    SCHEMA_VERSION,
    DatasetDNA,
    DatasetProfile,
    FeatureSchema,
    FinalTestResult,
    LeakageReport,
    ModelDecision,
    ProblemDefinition,
    QualityReport,
    ValidationPlan,
)
from mlverdict.data.dna import build_dna
from mlverdict.data.loader import load_dataset, require_target
from mlverdict.data.profiler import profile_dataset
from mlverdict.decision.record import build_decision_record
from mlverdict.decision.selector import select_model
from mlverdict.evaluation.evaluator import evaluate_experiments
from mlverdict.experiments.runner import run_experiments
from mlverdict.experiments.scoring import compute_metrics, extract_score
from mlverdict.metrics.intelligence import select_metric_plan
from mlverdict.models.candidates import select_candidates
from mlverdict.preprocessing.pipeline import build_pipeline
from mlverdict.preprocessing.planner import plan_preprocessing
from mlverdict.problem.detector import detect_problem
from mlverdict.production.artifact import ModelArtifact
from mlverdict.production.readiness import assess_readiness
from mlverdict.quality.analyzer import analyze_quality
from mlverdict.quality.leakage import detect_leakage
from mlverdict.reporting.report import render_report
from mlverdict.validation.splitter import isolate_final_test
from mlverdict.validation.strategy import select_validation_plan


class Verdict:
    """Evidence-based machine learning decision engine."""

    def __init__(
        self,
        random_state: int = 42,
        constraints: Constraints | dict[str, Any] | None = None,
        config: VerdictConfig | None = None,
        enable_hpo: bool = True,
        **config_overrides: Any,
    ) -> None:
        if config is None:
            config = VerdictConfig(random_state=random_state, **config_overrides)
        self.config = config
        self.enable_hpo = enable_hpo
        if constraints is None:
            self.constraints = Constraints()
        elif isinstance(constraints, Constraints):
            self.constraints = constraints
        else:
            self.constraints = Constraints(**constraints)

    def fit(
        self,
        data: str | pd.DataFrame,
        target: str,
        *,
        problem_type: str | ProblemType | None = None,
        group_col: str | None = None,
        time_col: str | None = None,
        primary_metric: str | None = None,
        validation_strategy: str | None = None,
        false_positive_cost: float | None = None,
        false_negative_cost: float | None = None,
        feature_columns: list[str] | None = None,
    ) -> Run:
        seed_everything(self.config.random_state)
        frame = load_dataset(data)
        require_target(frame, target)
        frame = frame.dropna(subset=[target]).reset_index(drop=True)

        provisional = _provisional_plan(frame, target, self.config, group_col, time_col, validation_strategy)
        split = isolate_final_test(frame, target, provisional, self.config.random_state)
        train = split.train
        test = split.test

        profile = profile_dataset(train, target, self.config)
        dna = build_dna(profile, self.config, group_col=group_col, time_col=time_col)
        problem = detect_problem(profile, train[target], user_problem_type=problem_type)
        quality = analyze_quality(train, profile, self.config)
        leakage = detect_leakage(train, profile, dna, target, group_col=group_col)

        if problem.status != DecisionStatus.DECIDED or problem.problem_type is None:
            return _early(
                DecisionStatus.UNDECIDED,
                profile,
                dna,
                problem,
                quality,
                leakage,
                notes=(
                    "Problem type is undecided. Pass problem_type to proceed.",
                    *problem.warnings,
                ),
            )

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
            )

        validation = select_validation_plan(
            dna,
            problem,
            self.config,
            group_col=group_col,
            time_col=time_col,
            user_strategy=validation_strategy,
        )
        metric_plan = select_metric_plan(
            dna,
            problem,
            user_metric=primary_metric,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        candidates = select_candidates(dna, problem, metric_plan, self.constraints)

        extra_drop = tuple(
            c
            for c in (group_col,)
            if c and c != target
        )
        feature_frame = _feature_frame(train, target, feature_columns, extra_drop)
        y = train[target]
        groups = train[group_col] if group_col and group_col in train.columns else None

        experiments = run_experiments(
            feature_frame,
            y,
            candidates.included,
            profile,
            dna,
            problem,
            metric_plan,
            validation,
            self.config,
            groups=groups,
            extra_drop=extra_drop,
            enable_hpo=self.enable_hpo,
        )
        evaluations = evaluate_experiments(experiments, self.config, self.constraints)
        assumptions = _assumptions(dna, validation, group_col, time_col)
        limitations = _limitations(quality, leakage, validation)
        decision = select_model(evaluations, metric_plan, assumptions=assumptions, limitations=limitations)

        final_test = None
        artifact = None
        production = None
        if decision.status == DecisionStatus.DECIDED and decision.selected_model:
            winner = next(c for c in candidates.included if c.name == decision.selected_model)
            winner_exp = next(e for e in experiments if e.model_name == winner.name)
            pre_plan = plan_preprocessing(profile, dna, winner.family, self.config, extra_drop=extra_drop)
            pipeline = build_pipeline(
                pre_plan,
                winner,
                problem.problem_type,
                self.config.random_state,
                winner_exp.configuration,
            )
            pipeline.fit(feature_frame, y)
            if len(test):
                X_test = _feature_frame(test, target, feature_columns, extra_drop)
                y_test = test[target]
                preds = pipeline.predict(X_test)
                proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
                test_metrics = compute_metrics(y_test, preds, proba, problem, metric_plan)
                primary = extract_score(test_metrics, metric_plan)
                raw_primary = test_metrics.get(metric_plan.primary.name, primary)
                gap = None
                if decision.selected_primary_score is not None:
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
                target_name=target,
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
                },
                metrics=final_test.metrics if final_test else winner_exp.metrics,
                decision_record={},  # filled after record is built
                configuration=asdict(self.config),
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
            constraints=asdict(self.constraints),
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
        )


def _feature_frame(
    frame: pd.DataFrame,
    target: str,
    feature_columns: list[str] | None,
    extra_drop: tuple[str, ...],
) -> pd.DataFrame:
    cols = list(feature_columns) if feature_columns else [c for c in frame.columns if c != target]
    drop = {target, *extra_drop}
    cols = [c for c in cols if c not in drop and c in frame.columns]
    return frame.loc[:, cols].copy()


def _provisional_plan(
    frame: pd.DataFrame,
    target: str,
    config: VerdictConfig,
    group_col: str | None,
    time_col: str | None,
    user_strategy: str | None,
) -> ValidationPlan:
    from mlverdict.core.enums import Confidence, DatasetScale, IIDAssumption

    n_unique = int(frame[target].nunique(dropna=True))
    classification = n_unique <= 20
    dummy_problem = ProblemDefinition(
        problem_type=ProblemType.BINARY_CLASSIFICATION if classification else ProblemType.REGRESSION,
        confidence=Confidence.MEDIUM,
        evidence=("provisional split only — not a modeling decision",),
        warnings=(),
        status=DecisionStatus.DECIDED,
    )
    dummy_dna = DatasetDNA(
        schema_version=SCHEMA_VERSION,
        n_rows=len(frame),
        n_features=max(frame.shape[1] - 1, 0),
        target_name=target,
        target_kind="categorical" if classification else "numeric",
        target_cardinality=n_unique,
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


def _assumptions(dna, validation, group_col, time_col) -> tuple[str, ...]:
    items = [
        f"IID assumption treated as {dna.iid_assumption.value}.",
        f"Validation strategy is {validation.strategy.value}.",
    ]
    if dna.entity_hint and not group_col:
        items.append(f"Entity hint '{dna.entity_hint}' was not confirmed as group_col.")
    if dna.time_hint and not time_col:
        items.append(f"Time hint '{dna.time_hint}' was not confirmed as time_col.")
    return tuple(items)


def _limitations(quality: QualityReport, leakage: LeakageReport, validation: ValidationPlan) -> tuple[str, ...]:
    items = [
        "Final test was isolated before modeling decisions and was scored once after selection.",
        "Leakage analysis reports signals, not a leakage-free certificate.",
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
    profile: DatasetProfile,
    dna: DatasetDNA,
    problem: ProblemDefinition,
    quality: QualityReport,
    leakage: LeakageReport,
    *,
    decision: ModelDecision | None = None,
    notes: tuple[str, ...] = (),
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
        report_text="MLVerdict " + status.value + "\n" + "\n".join(notes),
    )
