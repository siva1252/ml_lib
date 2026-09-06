"""Persist the full written judgment — foundation for later monitoring phases."""

from __future__ import annotations

from mlverdict.core.types import (
    SCHEMA_VERSION,
    CandidateSet,
    DatasetDNA,
    EvaluationResult,
    ExperimentResult,
    FinalTestResult,
    LeakageReport,
    MetricPlan,
    ModelDecision,
    ModelDecisionRecord,
    ProblemDefinition,
    ProductionReadinessResult,
    QualityReport,
    ValidationPlan,
)


def build_decision_record(
    *,
    dna: DatasetDNA,
    problem: ProblemDefinition,
    quality: QualityReport,
    leakage: LeakageReport,
    validation: ValidationPlan,
    metrics: MetricPlan,
    candidates: CandidateSet,
    experiments: tuple[ExperimentResult, ...],
    evaluations: tuple[EvaluationResult, ...],
    decision: ModelDecision,
    final_test: FinalTestResult | None,
    constraints: dict,
    production: ProductionReadinessResult | None,
    extra_limitations: tuple[str, ...] = (),
) -> ModelDecisionRecord:
    limitations = tuple(decision.limitations) + extra_limitations
    if leakage.signals:
        limitations = limitations + ("Leakage findings are signals only; the dataset is not certified clean.",)
    return ModelDecisionRecord(
        schema_version=SCHEMA_VERSION,
        dataset_dna=dna,
        problem=problem,
        quality=quality,
        leakage=leakage,
        validation=validation,
        metrics=metrics,
        candidates=tuple(c.name for c in candidates.included),
        excluded_candidates=tuple(f"{e.name}: {e.reason}" for e in candidates.excluded),
        experiments=experiments,
        evaluations=evaluations,
        decision=decision,
        final_test=final_test,
        constraints=constraints,
        production=production,
        limitations=limitations,
    )
