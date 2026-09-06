"""Run-level configuration. Intelligence proposes; this object records the law."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Constraints:
    """Hard production constraints. Failures reject a candidate, they do not get averaged away."""

    max_latency_ms: float | None = None
    max_train_time_seconds: float | None = None
    require_explainable: bool = False
    max_complexity: str | None = None  # low | medium | high


@dataclass(frozen=True)
class VerdictConfig:
    random_state: int = 42
    test_size: float = 0.2
    cv_splits: int = 5
    max_hpo_trials: int = 12
    max_hpo_time_seconds: float = 45.0
    n_hpo_candidates: int = 2
    near_constant_threshold: float = 0.95
    high_cardinality_threshold: int = 50
    id_unique_ratio: float = 0.98
    latency_probe_rows: int = 64
    min_rows_for_cv: int = 40
    min_rows_for_holdout: int = 20
    # Weights for soft ranking after hard constraints. Do not collapse quality to one ad-hoc number in callers.
    weight_performance: float = 0.45
    weight_stability: float = 0.25
    weight_generalization: float = 0.15
    weight_latency: float = 0.10
    weight_complexity: float = 0.05
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.05 <= self.test_size <= 0.4:
            raise ValueError("test_size must be between 0.05 and 0.4")
        if self.cv_splits < 2:
            raise ValueError("cv_splits must be >= 2")
        weights = (
            self.weight_performance
            + self.weight_stability
            + self.weight_generalization
            + self.weight_latency
            + self.weight_complexity
        )
        if abs(weights - 1.0) > 1e-6:
            raise ValueError("decision weights must sum to 1.0")
