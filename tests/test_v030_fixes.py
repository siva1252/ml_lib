"""Regression tests for the 0.3.0 audit fixes. Each test names the bug it guards against."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest

from mlverdict import ModelArtifact, Verdict
from mlverdict.core.enums import DecisionStatus, ProblemType, Severity
from mlverdict.core.exceptions import ArtifactError, ConfigurationError
from mlverdict.data.dna import build_dna
from mlverdict.data.profiler import profile_dataset
from mlverdict.evaluation.comparison import noise_aware_scale
from mlverdict.experiments import hpo
from mlverdict.models.adapters import available_boosters
from mlverdict.models.registry import CATALOG
from mlverdict.quality.leakage import detect_leakage


def _text_binary(n: int = 600, positive_share: float = 0.15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    plan = rng.choice(["basic", "pro", "ent"], size=n)
    score = 1.6 * x1 - 0.7 * x2 + (plan == "pro") * 0.5 + rng.normal(scale=0.6, size=n)
    cutoff = np.quantile(score, 1 - positive_share)
    return pd.DataFrame({"x1": x1, "x2": x2, "plan": plan, "y": np.where(score > cutoff, "churn", "stay")})


# --- Bug B: text labels made F1 / precision / recall / PR-AUC NaN for every model -------------


def test_text_labels_imbalanced_pr_auc_is_finite(fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(_text_binary(), "y")
    assert run.status == DecisionStatus.DECIDED
    assert run.metric_plan.primary.name == "pr_auc"
    assert run.problem.positive_label == "churn"  # minority class
    assert all(np.isfinite(e.mean_score) for e in run.experiments)
    for name in ("pr_auc", "f1", "precision", "recall", "roc_auc"):
        assert name in run.final_test.metrics, name


def test_text_labels_user_f1_is_finite(fast_config):
    frame = _text_binary(positive_share=0.4)
    run = Verdict(config=fast_config, enable_hpo=False).fit(frame, "y", primary_metric="f1")
    assert all(np.isfinite(e.mean_score) for e in run.experiments)
    assert 0.0 < run.final_test.metrics["f1"] <= 1.0


def test_positive_label_override_and_validation(fast_config):
    frame = _text_binary(positive_share=0.4)
    run = Verdict(config=fast_config, enable_hpo=False).fit(frame, "y", positive_label="stay")
    assert run.problem.positive_label == "stay"
    assert run.artifact().metadata["positive_label"] == "stay"
    with pytest.raises(ConfigurationError):
        Verdict(config=fast_config, enable_hpo=False).fit(frame, "y", positive_label="nope")


def test_experiment_without_finite_scores_is_failed_not_ranked(fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(_text_binary(), "y")
    ranked = {ev.model_name for ev in run.evaluations}
    for exp in run.experiments:
        if not np.isfinite(exp.mean_score):
            assert exp.model_name not in ranked


# --- Bug A: XGBoost rejected text labels ------------------------------------------------------


@pytest.mark.skipif(not available_boosters()["xgboost"], reason="xgboost not installed")
def test_xgboost_handles_text_labels(fast_config, monkeypatch):
    import mlverdict.api.verdict as verdict_mod
    from mlverdict.models import candidates as cand_mod

    original = cand_mod.select_candidates

    def only_xgb(*args, **kwargs):
        cs = original(*args, **kwargs)
        return type(cs)(included=tuple(c for c in cs.included if c.estimator_key == "xgboost"), excluded=cs.excluded)

    monkeypatch.setattr(verdict_mod, "select_candidates", only_xgb)
    run = Verdict(config=fast_config, enable_hpo=False).fit(_text_binary(), "y")
    assert run.status == DecisionStatus.DECIDED
    assert run.decision.selected_model == "XGBoost"
    preds = run.artifact().predict(_text_binary().drop(columns="y").head(5))
    assert set(preds) <= {"churn", "stay"}


# --- Model coverage ----------------------------------------------------------------------------


def test_catalog_has_classic_models():
    keys = {e["key"] for e in CATALOG}
    for key in ("linear_regression", "lasso", "elastic_net", "decision_tree", "knn", "svm", "gaussian_nb"):
        assert key in keys


@pytest.mark.parametrize(
    "key,kind",
    [
        ("decision_tree", "classification"),
        ("knn", "classification"),
        ("svm", "classification"),
        ("gaussian_nb", "classification"),
        ("linear_regression", "regression"),
        ("lasso", "regression"),
        ("elastic_net", "regression"),
        ("decision_tree", "regression"),
        ("knn", "regression"),
        ("svm", "regression"),
    ],
)
def test_each_new_model_trains_end_to_end(key, kind, fast_config, monkeypatch, tmp_path):
    import mlverdict.api.verdict as verdict_mod
    from mlverdict.models import candidates as cand_mod

    original = cand_mod.select_candidates

    def only(*args, **kwargs):
        cs = original(*args, **kwargs)
        return type(cs)(included=tuple(c for c in cs.included if c.estimator_key == key), excluded=cs.excluded)

    monkeypatch.setattr(verdict_mod, "select_candidates", only)
    if kind == "classification":
        frame, target = _text_binary(300), "y"
    else:
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
        frame["y"] = 3 * frame.a - frame.b + rng.normal(scale=0.3, size=300)
        target = "y"
    run = Verdict(config=fast_config).fit(frame, target)
    assert run.status == DecisionStatus.DECIDED, run.decision.reasons
    path = run.artifact().save(tmp_path / f"{key}.joblib")
    X = frame.drop(columns=target).head(5)
    assert list(ModelArtifact.load(path).predict(X)) == list(run.artifact().predict(X))


def test_svm_is_skipped_on_large_data(fast_config):
    from mlverdict.core.config import VerdictConfig

    config = VerdictConfig(cv_splits=3, max_hpo_trials=0, max_rows_for_svm=100)
    run = Verdict(config=config, enable_hpo=False).fit(_text_binary(400), "y")
    excluded = {x.name: x.reason for x in run.candidates.excluded}
    assert "Support Vector Machine" in excluded
    assert "max_rows_for_svm" in excluded["Support Vector Machine"]


# --- HPO crash when every trial fails ----------------------------------------------------------


def test_all_failed_hpo_trials_fall_back_to_defaults(fast_config, monkeypatch):
    def boom(trial, key):
        raise RuntimeError("boom")

    monkeypatch.setattr(hpo, "_space", boom)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    run = Verdict(config=fast_config).fit(_text_binary(), "y")
    assert run.status == DecisionStatus.DECIDED


# --- Robustness: inf values and bad prediction input ------------------------------------------


def test_inf_values_are_imputed_not_fatal(fast_config):
    frame = _text_binary()
    frame.loc[0, "x1"] = np.inf
    frame.loc[1, "x2"] = -np.inf
    run = Verdict(config=fast_config, enable_hpo=False).fit(frame, "y")
    assert run.status == DecisionStatus.DECIDED
    assert any(i.issue_type == "invalid_values" for i in run.quality.issues)


def test_artifact_rejects_text_in_numeric_column(fast_config):
    frame = _text_binary()
    artifact = Verdict(config=fast_config, enable_hpo=False).fit(frame, "y").artifact()
    bad = frame.drop(columns="y").head(3).copy()
    bad["x1"] = bad["x1"].astype(object)
    bad.loc[bad.index[0], "x1"] = "abc"
    with pytest.raises(ArtifactError, match="Non-numeric"):
        artifact.predict(bad)
    assert len(artifact.predict(bad, strict=False)) == 3
    ok = frame.drop(columns="y").head(3).astype({"x1": str})  # numbers sent as strings are fine
    assert len(artifact.predict(ok)) == 3


def test_readiness_checks_are_real_probes(fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(_text_binary(), "y")
    checks = {c.name: c for c in run.production.checks}
    assert "all-missing row" in checks["missing_values_handled"].detail
    assert "unseen category" in checks["unknown_categories_handled"].detail
    assert checks["dependencies_captured"].passed
    env = run.artifact().metadata["environment"]
    assert env["python"] and env["scikit-learn"]


# --- Leakage -----------------------------------------------------------------------------------


def test_numeric_copy_of_text_target_is_flagged():
    frame = _text_binary()
    frame["leaky"] = (frame.y == "churn").astype(float) + np.random.default_rng(0).normal(0, 1e-4, len(frame))
    profile = profile_dataset(frame, "y")
    report = detect_leakage(frame, profile, build_dna(profile), "y")
    hits = [s for s in report.signals if s.column == "leaky"]
    assert hits and hits[0].severity == Severity.HIGH_RISK


def test_days_since_feature_is_warning_not_high_risk():
    frame = _text_binary()
    frame["days_since_last_login"] = np.random.default_rng(0).integers(0, 90, len(frame))
    profile = profile_dataset(frame, "y")
    report = detect_leakage(frame, profile, build_dna(profile), "y")
    sig = [s for s in report.signals if s.column == "days_since_last_login"]
    assert sig and all(s.severity == Severity.WARNING for s in sig)


# --- Ranking -----------------------------------------------------------------------------------


def test_noise_aware_scale_does_not_inflate_tiny_gaps():
    scores = noise_aware_scale([0.9949, 0.9905], [0.004, 0.005], [5, 5])
    assert scores[0] == 1.0
    assert scores[1] > 0.4  # plain min-max would give 0.0
    spread = noise_aware_scale([0.95, 0.60], [0.01, 0.01], [5, 5])
    assert spread[1] == 0.0


# --- Problem detection -------------------------------------------------------------------------


def test_integer_class_codes_are_multiclass_but_float_ratings_stay_undecided(fast_config):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(240, 2))
    codes = rng.integers(0, 3, size=240)
    frame = pd.DataFrame({"a": x[:, 0] + codes, "b": x[:, 1], "label": codes})
    run = Verdict(config=fast_config, enable_hpo=False).fit(frame, "label")
    assert run.problem.problem_type == ProblemType.MULTICLASS_CLASSIFICATION
    frame["label"] = (codes + 1).astype(float)
    assert Verdict(config=fast_config, enable_hpo=False).fit(frame, "label").status == DecisionStatus.UNDECIDED


# --- Unsupervised quality ----------------------------------------------------------------------


def test_anomaly_detection_flags_about_the_configured_share(fast_config):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(1200, 4)), columns=list("abcd"))
    planted = rng.choice(len(frame), 24, replace=False)
    frame.iloc[planted] += 8
    run = Verdict(config=fast_config).fit(frame, task="anomaly_detection")
    pred = run.artifact().predict(frame)
    flagged = np.where(pred == -1)[0]
    assert len(set(flagged) & set(planted)) >= 22
    assert len(flagged) <= 0.08 * len(frame)  # policy is 5%; old metric flagged ~29%


def test_pca_keeps_only_the_real_dimensions(fast_config):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(600, 3))
    frame = pd.DataFrame(base @ rng.normal(size=(3, 12)) + rng.normal(0, 0.05, (600, 12))).add_prefix("f")
    run = Verdict(config=fast_config).fit(frame, task="dimensionality_reduction")
    z = run.artifact().predict(frame)
    assert np.asarray(z).shape[1] <= 4


def test_clustering_large_data_path_samples_and_finds_k():
    """k is chosen on a sample (cluster_scan_rows) and silhouette is sampled; both must stay correct."""
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    from mlverdict.core.config import VerdictConfig

    X, y = make_blobs(6000, n_features=4, centers=5, random_state=0)
    frame = pd.DataFrame(X, columns=list("abcd"))
    config = VerdictConfig(cv_splits=3, cluster_scan_rows=1500, silhouette_sample_size=1000, max_clusters=8)
    run = Verdict(config=config, enable_hpo=False).fit(frame, task="clustering")
    assert run.status == DecisionStatus.DECIDED
    labels = run.artifact().predict(frame)
    assert adjusted_rand_score(y, labels) > 0.95


def test_positive_label_rejected_for_unsupervised(fast_config):
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 3)), columns=list("abc"))
    with pytest.raises(ConfigurationError):
        Verdict(config=fast_config).fit(frame, task="clustering", positive_label="x")


# --- Robustness pass -----------------------------------------------------------------------------


def test_mixed_type_column_is_treated_as_category(fast_config):
    frame = _text_binary(300).assign(mix=[1, "a", 2.5, None, True, "b"] * 50)
    run = Verdict(config=fast_config, enable_hpo=False).fit(frame, "y")
    assert run.status == DecisionStatus.DECIDED
    assert len(run.artifact().predict(frame.drop(columns="y").head(6))) == 6


def test_duplicate_column_names_raise_clear_error(fast_config):
    from mlverdict.core.exceptions import DataLoadError

    frame = pd.concat([_text_binary(100), _text_binary(100)[["x1"]]], axis=1)
    with pytest.raises(DataLoadError, match="Duplicate column names"):
        Verdict(config=fast_config).fit(frame, "y")


def test_predict_on_empty_frame_returns_empty(fast_config):
    frame = _text_binary(300)
    artifact = Verdict(config=fast_config, enable_hpo=False).fit(frame, "y").artifact()
    empty = frame.drop(columns="y").head(0)
    assert len(artifact.predict(empty)) == 0
    assert artifact.predict_proba(empty).shape == (0, 2)


def test_integer_column_names(fast_config):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(300, 4)))
    frame["y"] = np.where(frame[0] + frame[1] > 0, "a", "b")
    run = Verdict(config=fast_config, enable_hpo=False).fit(frame, "y")
    assert run.status == DecisionStatus.DECIDED
    assert len(run.artifact().predict(frame.drop(columns="y").head(3))) == 3
