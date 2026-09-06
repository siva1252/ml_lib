import pandas as pd
import pytest

from mlverdict import DecisionStatus, Verdict
from mlverdict.core.exceptions import DataLoadError, TargetNotFoundError
from mlverdict.production.artifact import ModelArtifact


def test_missing_target_raises(binary_frame, fast_config):
    with pytest.raises(TargetNotFoundError):
        Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, "not_a_column")


def test_empty_dataframe_raises(fast_config):
    with pytest.raises(DataLoadError):
        Verdict(config=fast_config).fit(pd.DataFrame(), "y")


def test_missing_file_raises(fast_config):
    with pytest.raises(DataLoadError):
        Verdict(config=fast_config).fit("definitely_missing_file.csv", "churn")


def test_ambiguous_target_does_not_silently_continue(ambiguous_target_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(ambiguous_target_frame, "score")
    assert run.status == DecisionStatus.UNDECIDED
    assert run.best() is None
    assert run.artifact() is None
    assert run.experiments == ()


def test_quality_blocker_on_tiny_constant_data(fast_config):
    frame = pd.DataFrame({"x": [1, 1, 1, 1], "y": [0, 0, 0, 0]})
    run = Verdict(config=fast_config, enable_hpo=False).fit(
        frame, "y", problem_type="binary_classification"
    )
    assert run.status in {DecisionStatus.BLOCKED, DecisionStatus.UNDECIDED}


def test_unknown_metric_raises(binary_frame, fast_config):
    from mlverdict.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        Verdict(config=fast_config, enable_hpo=False).fit(
            binary_frame, "churn", primary_metric="not_a_real_metric"
        )


def test_artifact_missing_features(binary_frame, fast_config):
    from mlverdict.core.exceptions import ArtifactError

    run = Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, "churn")
    with pytest.raises(ArtifactError):
        run.artifact().predict(pd.DataFrame({"nope": [1, 2, 3]}))


def test_artifact_missing_path():
    from mlverdict.core.exceptions import ArtifactError

    with pytest.raises(ArtifactError):
        ModelArtifact.load("no_such_artifact.joblib")
