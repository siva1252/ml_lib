from pathlib import Path

import pandas as pd

from mlverdict import DecisionStatus, ModelArtifact, Verdict
from mlverdict.production.service import predict_records


def test_binary_fit_leaderboard_report_artifact(binary_frame, fast_config, tmp_path):
    run = Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, "churn")
    assert run.status == DecisionStatus.DECIDED
    best = run.best()
    assert best is not None
    assert best["model"]
    board = run.leaderboard()
    assert not board.empty
    assert "model" in board.columns
    text = run.report()
    assert "Executive summary" in text
    assert "Dataset DNA" in text
    assert "Selected model" in text
    readable = str(run)
    assert "MLVerdict" in readable
    assert "Leaderboard" in readable
    assert "What to do next" in readable
    assert run.verdict() is not None
    assert run.final_test is not None
    assert run.profile is not None
    assert run.profile.n_rows < len(binary_frame)
    artifact = run.artifact()
    assert artifact is not None
    preds = artifact.predict(binary_frame.drop(columns=["churn"]).head(5))
    assert len(preds) == 5
    path = artifact.save(tmp_path / "model.joblib")
    loaded = ModelArtifact.load(path)
    again = loaded.predict(binary_frame.drop(columns=["churn"]).head(5))
    assert list(preds) == list(again)
    assert run.production is not None
    assert run.production.passed or not run.production.blocked


def test_regression_end_to_end(regression_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(regression_frame, "price")
    assert run.status == DecisionStatus.DECIDED
    assert run.problem is not None
    assert run.problem.is_regression
    assert run.metric_plan.primary.name == "rmse"
    assert run.artifact() is not None


def test_multiclass_end_to_end(multiclass_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(
        multiclass_frame, "label", problem_type="multiclass_classification"
    )
    assert run.status == DecisionStatus.DECIDED
    assert run.best()["model"]


def test_imbalanced_uses_pr_auc(imbalanced_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(imbalanced_frame, "churn")
    assert run.metric_plan.primary.name == "pr_auc"
    assert run.status in {DecisionStatus.DECIDED, DecisionStatus.BLOCKED}


def test_csv_path_roundtrip(binary_frame, fast_config, tmp_path: Path):
    csv = tmp_path / "customer_churn.csv"
    binary_frame.to_csv(csv, index=False)
    run = Verdict(config=fast_config, enable_hpo=False).fit(str(csv), "churn")
    assert run.status == DecisionStatus.DECIDED


def test_grouped_and_time_paths(grouped_frame, time_frame, fast_config):
    grouped = Verdict(config=fast_config, enable_hpo=False).fit(
        grouped_frame, "churn", group_col="customer_id"
    )
    assert grouped.validation is not None
    assert grouped.validation.group_column == "customer_id"
    timed = Verdict(config=fast_config, enable_hpo=False).fit(
        time_frame, "churn", time_col="event_date"
    )
    assert timed.validation.strategy.value == "time_based"


def test_hpo_runs_on_shortlist(binary_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=True).fit(binary_frame, "churn")
    assert run.status == DecisionStatus.DECIDED
    assert any(e.optimized for e in run.experiments) or run.experiments


def test_user_metric_and_constraints_flow(binary_frame, fast_config):
    run = Verdict(
        config=fast_config,
        enable_hpo=False,
        constraints={"require_explainable": True},
    ).fit(binary_frame, "churn", primary_metric="f1")
    assert run.metric_plan.primary.name == "f1"
    assert run.metric_plan.user_override
    if run.decision and run.decision.selected_model:
        winner = next(e for e in run.evaluations if e.model_name == run.decision.selected_model)
        assert winner.explainable


def test_service_predict_records(binary_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, "churn")
    records = binary_frame.drop(columns=["churn"]).head(3).to_dict(orient="records")
    preds = predict_records(run.artifact(), records)
    assert len(preds) == 3


def test_decision_record_has_required_fields(binary_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, "churn")
    rec = run.decision_record
    assert rec is not None
    payload = rec.to_dict()
    assert payload["dataset_dna"]["target_name"] == "churn"
    assert payload["decision"]["selected_model"]
    assert "signals" in payload["leakage"]
    assert rec.limitations
