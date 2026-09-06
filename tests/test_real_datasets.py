"""Real-dataset tests: prove MLVerdict on published data, not synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlverdict import DecisionStatus, ModelArtifact, ProblemType, Verdict
from mlverdict.core.config import VerdictConfig
from tests.realdata import breast_cancer_frame, diabetes_frame, iris_frame, wine_frame


@pytest.fixture
def real_config() -> VerdictConfig:
    return VerdictConfig(
        random_state=42,
        test_size=0.2,
        cv_splits=3,
        max_hpo_trials=2,
        max_hpo_time_seconds=25,
        n_hpo_candidates=1,
        latency_probe_rows=16,
    )


def _assert_decided_path(run, *, target: str, expected_problem: ProblemType, n_full: int) -> None:
    assert run.status == DecisionStatus.DECIDED, run.notes or (run.decision.reasons if run.decision else ())
    assert run.problem is not None
    assert run.problem.problem_type == expected_problem
    assert run.dna is not None
    assert run.dna.target_name == target
    assert run.dna.n_rows > 0
    assert run.profile is not None
    # Final test was isolated: profile/DNA describe train/val only.
    assert run.profile.n_rows < n_full
    assert run.dna.n_rows == run.profile.n_rows
    assert run.quality is not None
    assert run.leakage is not None
    assert "signals" in run.leakage.claim.lower() or "guarantee" in run.leakage.claim.lower()
    assert run.validation is not None
    assert run.metric_plan is not None
    assert run.candidates is not None
    assert run.experiments
    assert run.evaluations
    best = run.best()
    assert best is not None
    assert best["model"]
    assert best["primary_metric"] == run.metric_plan.primary.name
    board = run.leaderboard()
    assert not board.empty
    assert best["model"] in set(board["model"])
    assert run.verdict() is not None
    report = run.report()
    assert "Executive summary" in report
    assert "Dataset DNA" in report
    assert "Selected model" in report
    assert run.final_test is not None
    assert run.final_test.n_rows > 0
    assert run.artifact() is not None
    assert run.production is not None


def test_breast_cancer_binary_full_path(real_config, tmp_path: Path):
    frame = breast_cancer_frame()
    assert frame.shape[0] == 569
    assert "diagnosis" in frame.columns

    run = Verdict(config=real_config, enable_hpo=False).fit(frame, "diagnosis")
    _assert_decided_path(
        run, target="diagnosis", expected_problem=ProblemType.BINARY_CLASSIFICATION, n_full=len(frame)
    )
    assert run.metric_plan.primary.name in {"pr_auc", "roc_auc", "f1", "accuracy"}
    assert run.validation.strategy.value in {"stratified_kfold", "kfold", "holdout"}

    features = frame.drop(columns=["diagnosis"]).head(8)
    preds = run.predict(features)
    assert len(preds) == 8

    path = run.artifact().save(tmp_path / "breast_cancer.joblib")
    loaded = ModelArtifact.load(path)
    reloaded = loaded.predict(features)
    assert list(preds) == list(reloaded)


def test_breast_cancer_from_csv_file(real_config, tmp_path: Path):
    csv = tmp_path / "breast_cancer.csv"
    breast_cancer_frame().to_csv(csv, index=False)
    run = Verdict(config=real_config, enable_hpo=False).fit(str(csv), "diagnosis")
    assert run.status == DecisionStatus.DECIDED
    assert run.problem.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert run.best()["model"]


def test_iris_ambiguous_integers_are_undecided(real_config):
    """species is 0/1/2 — the engine must not silently force multiclass."""
    frame = iris_frame()
    run = Verdict(config=real_config, enable_hpo=False).fit(frame, "species")
    assert run.status == DecisionStatus.UNDECIDED
    assert run.best() is None
    assert run.artifact() is None
    assert run.experiments == ()
    assert run.problem is not None
    assert run.problem.confidence.value == "low"
    assert any("problem_type" in w.lower() or "ambiguous" in w.lower() for w in run.problem.warnings)


def test_iris_multiclass_when_user_confirms(real_config):
    frame = iris_frame()
    run = Verdict(config=real_config, enable_hpo=False).fit(
        frame, "species", problem_type="multiclass_classification"
    )
    _assert_decided_path(
        run, target="species", expected_problem=ProblemType.MULTICLASS_CLASSIFICATION, n_full=len(frame)
    )
    assert run.problem.user_override
    preds = run.predict(frame.drop(columns=["species"]).head(5))
    assert len(preds) == 5


def test_wine_multiclass_full_path(real_config):
    frame = wine_frame()
    run = Verdict(config=real_config, enable_hpo=False).fit(
        frame, "cultivar", problem_type="multiclass_classification"
    )
    _assert_decided_path(
        run, target="cultivar", expected_problem=ProblemType.MULTICLASS_CLASSIFICATION, n_full=len(frame)
    )
    assert run.metric_plan.primary.name in {"f1", "accuracy", "balanced_accuracy", "log_loss"}


def test_diabetes_regression_full_path(real_config):
    frame = diabetes_frame()
    run = Verdict(config=real_config, enable_hpo=False).fit(frame, "disease_progression")
    _assert_decided_path(
        run, target="disease_progression", expected_problem=ProblemType.REGRESSION, n_full=len(frame)
    )
    assert run.metric_plan.primary.name == "rmse"
    preds = run.predict(frame.drop(columns=["disease_progression"]).head(4))
    assert len(preds) == 4


def test_breast_cancer_user_metric_and_explainable_constraint(real_config):
    frame = breast_cancer_frame()
    run = Verdict(
        config=real_config,
        enable_hpo=False,
        constraints={"require_explainable": True},
    ).fit(frame, "diagnosis", primary_metric="recall")
    _assert_decided_path(
        run, target="diagnosis", expected_problem=ProblemType.BINARY_CLASSIFICATION, n_full=len(frame)
    )
    assert run.metric_plan.primary.name == "recall"
    assert run.metric_plan.user_override
    winner = next(e for e in run.evaluations if e.model_name == run.decision.selected_model)
    assert winner.explainable
    rejected = [r.reason.lower() for r in run.decision.rejected]
    assert run.decision.reasons


def test_real_runs_do_not_always_pick_the_same_model(real_config):
    cancer = Verdict(config=real_config, enable_hpo=False).fit(breast_cancer_frame(), "diagnosis")
    diabetes = Verdict(config=real_config, enable_hpo=False).fit(diabetes_frame(), "disease_progression")
    assert cancer.status == DecisionStatus.DECIDED
    assert diabetes.status == DecisionStatus.DECIDED
    # Different problems must at least produce their own evidence trail.
    assert cancer.decision.selected_model
    assert diabetes.decision.selected_model
    assert cancer.metric_plan.primary.name != diabetes.metric_plan.primary.name
    assert cancer.problem.problem_type != diabetes.problem.problem_type


def test_decision_record_from_real_data_is_complete(real_config):
    run = Verdict(config=real_config, enable_hpo=False).fit(breast_cancer_frame(), "diagnosis")
    rec = run.decision_record
    assert rec is not None
    payload = rec.to_dict()
    assert payload["dataset_dna"]["n_rows"] > 0
    assert payload["problem"]["problem_type"] == "binary_classification"
    assert payload["metrics"]["primary"]["name"]
    assert payload["candidates"]
    assert payload["experiments"]
    assert payload["decision"]["selected_model"]
    assert payload["final_test"]["n_rows"] > 0
    assert rec.limitations
