"""Stable contracts between modules. Add fields per milestone; do not pass raw dicts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mlverdict.core.enums import (
    ColumnRole,
    Confidence,
    DatasetScale,
    DecisionStatus,
    ExperimentStatus,
    IIDAssumption,
    ModelFamily,
    ProblemType,
    Severity,
    ValidationStrategy,
)

SCHEMA_VERSION = "1.0"


def _as_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    role: ColumnRole
    n_unique: int
    unique_ratio: float
    n_missing: int
    missing_ratio: float
    is_constant: bool
    is_near_constant: bool
    is_potential_id: bool
    sample_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetProfile:
    name: str
    dtype: str
    n_unique: int
    unique_ratio: float
    n_missing: int
    class_counts: tuple[tuple[str, int], ...] | None = None
    imbalance_ratio: float | None = None
    mean: float | None = None
    std: float | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class DatasetProfile:
    schema_version: str
    n_rows: int
    n_columns: int
    n_duplicate_rows: int
    columns: tuple[ColumnProfile, ...]
    target: TargetProfile
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]

    def column(self, name: str) -> ColumnProfile | None:
        for col in self.columns:
            if col.name == name:
                return col
        return None


@dataclass(frozen=True)
class DatasetDNA:
    schema_version: str
    n_rows: int
    n_features: int
    target_name: str
    target_kind: str
    target_cardinality: int
    imbalance_ratio: float | None
    missingness: str
    has_duplicates: bool
    has_datetime: bool
    has_potential_ids: bool
    potential_id_columns: tuple[str, ...]
    high_cardinality_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    near_constant_columns: tuple[str, ...]
    entity_hint: str | None
    time_hint: str | None
    iid_assumption: IIDAssumption
    scale: DatasetScale
    numeric_fraction: float
    categorical_fraction: float


@dataclass(frozen=True)
class ProblemDefinition:
    problem_type: ProblemType | None
    confidence: Confidence
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    status: DecisionStatus
    user_override: bool = False

    @property
    def is_classification(self) -> bool:
        return self.problem_type in {
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        }

    @property
    def is_regression(self) -> bool:
        return self.problem_type == ProblemType.REGRESSION

    @property
    def is_unsupervised(self) -> bool:
        return self.problem_type in {
            ProblemType.CLUSTERING,
            ProblemType.ANOMALY_DETECTION,
            ProblemType.DIMENSIONALITY_REDUCTION,
        }


@dataclass(frozen=True)
class QualityIssue:
    issue_type: str
    severity: Severity
    column: str | None
    evidence: str
    description: str
    recommendation: str


@dataclass(frozen=True)
class QualityReport:
    issues: tuple[QualityIssue, ...]

    @property
    def blockers(self) -> tuple[QualityIssue, ...]:
        return tuple(i for i in self.issues if i.severity == Severity.BLOCKER)

    @property
    def high_risks(self) -> tuple[QualityIssue, ...]:
        return tuple(i for i in self.issues if i.severity == Severity.HIGH_RISK)


@dataclass(frozen=True)
class LeakageSignal:
    signal_type: str
    severity: Severity
    column: str | None
    evidence: str
    description: str
    recommendation: str


@dataclass(frozen=True)
class LeakageReport:
    signals: tuple[LeakageSignal, ...]
    claim: str = "These are leakage *signals*, not a guarantee that the dataset is leakage-free."


@dataclass(frozen=True)
class ValidationPlan:
    strategy: ValidationStrategy
    n_splits: int
    test_size: float
    group_column: str | None
    time_column: str | None
    stratify: bool
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    shuffle: bool = True


@dataclass(frozen=True)
class MetricSpec:
    name: str
    greater_is_better: bool
    needs_proba: bool = False
    sklearn_name: str | None = None


@dataclass(frozen=True)
class MetricPlan:
    primary: MetricSpec
    secondary: tuple[MetricSpec, ...]
    evidence: tuple[str, ...]
    user_override: bool = False

    @property
    def all_metrics(self) -> tuple[MetricSpec, ...]:
        seen = {self.primary.name}
        extra = tuple(m for m in self.secondary if m.name not in seen)
        return (self.primary,) + extra


@dataclass(frozen=True)
class CandidateModel:
    name: str
    family: ModelFamily
    estimator_key: str
    why_included: str
    params: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    explainable: bool = False
    complexity: str = "medium"


@dataclass(frozen=True)
class ExclusionRecord:
    name: str
    reason: str


@dataclass(frozen=True)
class CandidateSet:
    included: tuple[CandidateModel, ...]
    excluded: tuple[ExclusionRecord, ...]


@dataclass(frozen=True)
class PreprocessPlan:
    numeric_imputation: str
    categorical_imputation: str
    numeric_scaling: bool
    categorical_encoding: str
    datetime_features: bool
    handle_unknown: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    estimator_key: str
    configuration: dict[str, Any]
    metrics: dict[str, float]
    fold_scores: tuple[float, ...]
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    train_score: float | None
    generalization_gap: float | None
    train_time_seconds: float
    infer_latency_ms: float
    status: ExperimentStatus
    error: str | None
    n_samples: int
    n_features: int
    validation_strategy: str
    optimized: bool
    family: str
    explainable: bool
    complexity: str


@dataclass(frozen=True)
class EvaluationResult:
    model_name: str
    predictive: dict[str, float]
    primary_score: float
    stability_score: float
    generalization_score: float
    latency_ms: float
    train_time_seconds: float
    complexity: str
    explainable: bool
    constraint_failures: tuple[str, ...]
    passes_constraints: bool
    composite_score: float
    fold_scores: tuple[float, ...]


@dataclass(frozen=True)
class Rejection:
    model_name: str
    reason: str


@dataclass(frozen=True)
class ModelDecision:
    status: DecisionStatus
    selected_model: str | None
    reasons: tuple[str, ...]
    tradeoffs: tuple[str, ...]
    rejected: tuple[Rejection, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    primary_metric: str | None = None
    selected_primary_score: float | None = None


@dataclass(frozen=True)
class FinalTestResult:
    metrics: dict[str, float]
    primary_score: float
    validation_primary_score: float | None
    gap: float | None
    n_rows: int


@dataclass(frozen=True)
class ProductionCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ProductionReadinessResult:
    passed: bool
    blocked: bool
    checks: tuple[ProductionCheck, ...]


@dataclass(frozen=True)
class ModelDecisionRecord:
    schema_version: str
    dataset_dna: DatasetDNA
    problem: ProblemDefinition
    quality: QualityReport
    leakage: LeakageReport
    validation: ValidationPlan
    metrics: MetricPlan
    candidates: tuple[str, ...]
    excluded_candidates: tuple[str, ...]
    experiments: tuple[ExperimentResult, ...]
    evaluations: tuple[EvaluationResult, ...]
    decision: ModelDecision
    final_test: FinalTestResult | None
    constraints: dict[str, Any]
    production: ProductionReadinessResult | None
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)


@dataclass(frozen=True)
class FeatureSchema:
    feature_names: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    target_name: str
    problem_type: str
    dtypes: tuple[tuple[str, str], ...]
