"""The object a user holds after fit(): inspect, explain, export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mlverdict.core.enums import DecisionStatus
from mlverdict.core.types import (
    CandidateSet,
    DatasetDNA,
    DatasetProfile,
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
from mlverdict.production.artifact import ModelArtifact
from mlverdict.reporting.display import render_console, render_summary
from mlverdict.reporting.report import render_report


@dataclass
class Run:
    status: DecisionStatus
    profile: DatasetProfile | None = None
    dna: DatasetDNA | None = None
    problem: ProblemDefinition | None = None
    quality: QualityReport | None = None
    leakage: LeakageReport | None = None
    validation: ValidationPlan | None = None
    metric_plan: MetricPlan | None = None
    candidates: CandidateSet | None = None
    experiments: tuple[ExperimentResult, ...] = ()
    evaluations: tuple[EvaluationResult, ...] = ()
    decision: ModelDecision | None = None
    decision_record: ModelDecisionRecord | None = None
    final_test: FinalTestResult | None = None
    production: ProductionReadinessResult | None = None
    artifact_obj: ModelArtifact | None = None
    report_text: str = ""
    notes: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return render_console(self)

    def __repr__(self) -> str:
        model = self.decision.selected_model if self.decision else None
        metric = self.decision.primary_metric if self.decision else None
        return (
            f"Run(status={self.status.value}, model={model!r}, "
            f"metric={metric!r})"
        )

    def display(self) -> str:
        """Print the human-readable verdict and return the same text."""
        text = str(self)
        print(text)
        return text

    def summary(self) -> str:
        return render_summary(self)

    def best(self) -> dict[str, Any] | None:
        if self.decision is None or self.decision.selected_model is None:
            return None
        ev = next((e for e in self.evaluations if e.model_name == self.decision.selected_model), None)
        return {
            "model": self.decision.selected_model,
            "status": self.status.value,
            "primary_metric": self.decision.primary_metric,
            "validation_score": self.decision.selected_primary_score,
            "final_test_score": self.final_test.primary_score if self.final_test else None,
            "latency_ms": ev.latency_ms if ev else None,
            "stability": ev.stability_score if ev else None,
            "reasons": list(self.decision.reasons),
        }

    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for ev in self.evaluations:
            rows.append(
                {
                    "model": ev.model_name,
                    "primary": ev.primary_score,
                    "stability": ev.stability_score,
                    "generalization": ev.generalization_score,
                    "latency_ms": ev.latency_ms,
                    "complexity": ev.complexity,
                    "passes_constraints": ev.passes_constraints,
                    "composite": ev.composite_score,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["passes_constraints", "composite"], ascending=[False, False]).reset_index(drop=True)

    def verdict(self) -> ModelDecision | ModelDecisionRecord | None:
        return self.decision_record or self.decision

    def report(self) -> str:
        if self.report_text:
            return self.report_text
        if self.decision_record is not None:
            return render_report(self.decision_record)
        return f"MLVerdict status: {self.status.value}\n" + "\n".join(self.notes)

    def artifact(self) -> ModelArtifact | None:
        return self.artifact_obj

    def predict(self, data: pd.DataFrame | list[dict[str, Any]]):
        if self.artifact_obj is None:
            raise RuntimeError("No deployable artifact. The run did not reach a ready model.")
        return self.artifact_obj.predict(data)
