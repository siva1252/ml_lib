import numpy as np
import pandas as pd
import pytest

from mlverdict import DecisionStatus, UnsupervisedTask, Verdict
from mlverdict.core.exceptions import ConfigurationError


def _features_only() -> pd.DataFrame:
    return pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "group": ["a", "b", "a", "b", "a", "b"]})


def _blob_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    half = n // 2
    a = rng.normal(size=(half, 2))
    b = rng.normal(loc=6.0, size=(n - half, 2))
    return pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])


def _anomaly_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    normal = rng.normal(size=(n - 12, 2))
    outliers = rng.normal(loc=8.0, scale=0.4, size=(12, 2))
    return pd.DataFrame(np.vstack([normal, outliers]), columns=["x", "y"])


def _pca_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    z = rng.normal(size=n)
    return pd.DataFrame(
        {
            "a": z + rng.normal(scale=0.05, size=n),
            "b": 2 * z + rng.normal(scale=0.05, size=n),
            "c": rng.normal(size=n),
            "d": -z + rng.normal(scale=0.05, size=n),
        }
    )


def test_target_positional_is_supervised(binary_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, "churn")
    assert run.status == DecisionStatus.DECIDED
    assert run.best() is not None
    assert run.artifact() is not None
    assert len(run.experiments) > 0
    assert run.extras.get("learning_mode") is None


def test_target_keyword_is_supervised(binary_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, target="churn")
    assert run.status == DecisionStatus.DECIDED
    assert run.best() is not None
    assert not run.leaderboard().empty


def test_omitted_target_does_not_train_supervised(fast_config):
    run = Verdict(config=fast_config, enable_hpo=True).fit(_features_only())
    assert run.experiments == ()
    assert run.evaluations == ()
    assert run.best() is None
    assert run.artifact() is None
    assert run.leaderboard().empty
    assert run.final_test is None


def test_omitted_target_and_task_is_undecided(fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(_features_only())
    assert run.status == DecisionStatus.UNDECIDED
    blob = " ".join(run.notes).lower()
    assert "no target" in blob
    assert "clustering" in blob
    assert run.extras.get("learning_mode") == "unsupervised_candidate"
    assert run.extras.get("unsupervised_task") is None


@pytest.mark.parametrize(
    "task",
    [
        "clustering",
        UnsupervisedTask.CLUSTERING,
    ],
)
def test_explicit_clustering_runs(fast_config, task):
    run = Verdict(config=fast_config, enable_hpo=False).fit(_blob_frame(), task=task)
    expected = task.value if isinstance(task, UnsupervisedTask) else task
    assert run.status == DecisionStatus.DECIDED
    assert run.extras.get("unsupervised_task") == expected
    assert run.extras.get("learning_mode") == "unsupervised"
    assert len(run.experiments) > 0
    assert run.best() is not None
    assert run.artifact() is not None
    assert run.metric_plan is not None
    assert run.metric_plan.primary.name == "silhouette"
    preds = run.artifact().predict(_blob_frame().head(8))
    assert len(preds) == 8


def test_anomaly_detection_runs(fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(_anomaly_frame(), task="anomaly_detection")
    assert run.status == DecisionStatus.DECIDED
    assert run.extras.get("unsupervised_task") == "anomaly_detection"
    assert run.best() is not None
    assert run.artifact() is not None
    assert "decision_std" in (run.best() or {}).get("primary_metric", "decision_std")
    labels = run.artifact().predict(_anomaly_frame().head(10))
    assert len(labels) == 10


def test_dimensionality_reduction_runs(fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(_pca_frame(), task="dimensionality_reduction")
    assert run.status == DecisionStatus.DECIDED
    assert run.extras.get("unsupervised_task") == "dimensionality_reduction"
    assert run.artifact() is not None
    z = run.artifact().predict(_pca_frame().head(5))
    assert len(z) == 5
    assert run.metric_plan.primary.name == "explained_variance"


def test_unsupervised_metrics_are_not_labeled_accuracy(fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(_blob_frame(), task="clustering")
    assert run.status == DecisionStatus.DECIDED
    text = run.report().lower()
    assert "silhouette" in text
    names = set()
    for exp in run.experiments:
        names.update(exp.metrics)
    assert "accuracy" not in names
    assert "roc_auc" not in names
    assert "silhouette" in names or run.metric_plan.primary.name == "silhouette"


def test_tiny_unsupervised_is_quality_blocked_not_fake_halt(fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(_features_only(), task="clustering")
    assert run.status == DecisionStatus.BLOCKED
    assert run.extras.get("unsupervised_task") == "clustering"
    blob = " ".join(run.notes).lower()
    assert "small" in blob or "few" in blob
    assert "not implemented" not in blob


def test_invalid_task_is_rejected(fast_config):
    with pytest.raises(ConfigurationError, match="Unknown task"):
        Verdict(config=fast_config).fit(_features_only(), task="forecasting")


def test_target_and_task_together_rejected(binary_frame, fast_config):
    with pytest.raises(ConfigurationError, match="not both"):
        Verdict(config=fast_config).fit(binary_frame, target="churn", task="clustering")
