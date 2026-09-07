"""Shared enumerations — the vocabulary every module uses."""

from __future__ import annotations

from enum import Enum


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


class DecisionStatus(str, Enum):
    DECIDED = "DECIDED"
    UNDECIDED = "UNDECIDED"
    BLOCKED = "BLOCKED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    HIGH_RISK = "HIGH_RISK"
    BLOCKER = "BLOCKER"


class ValidationStrategy(str, Enum):
    KFOLD = "kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    GROUP_KFOLD = "group_kfold"
    GROUP_SHUFFLE_SPLIT = "group_shuffle_split"
    TIME_BASED = "time_based"
    HOLDOUT = "holdout"


class ColumnRole(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    IDENTIFIER = "identifier"
    TEXT = "text"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


class ModelFamily(str, Enum):
    LINEAR = "linear"
    TREE = "tree"
    BOOSTING = "boosting"
    UNSUPERVISED = "unsupervised"


class ExperimentStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IIDAssumption(str, Enum):
    LIKELY = "likely"
    DOUBTFUL = "doubtful"
    UNLIKELY = "unlikely"


class DatasetScale(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class UnsupervisedTask(str, Enum):
    """Explicit unsupervised objectives. Missing a target is not enough to pick one."""

    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
