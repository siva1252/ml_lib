import pandas as pd

from mlverdict.core.enums import ModelFamily, ProblemType
from mlverdict.core.types import CandidateModel
from mlverdict.data.dna import build_dna
from mlverdict.data.profiler import profile_dataset
from mlverdict.preprocessing.pipeline import build_pipeline
from mlverdict.preprocessing.planner import plan_preprocessing


def test_pipeline_handles_missing_and_unknown_categories():
    train = pd.DataFrame(
        {
            "num": [1.0, 2.0, None, 4.0],
            "cat": ["a", "b", "a", "b"],
            "y": [0, 1, 0, 1],
        }
    )
    profile = profile_dataset(train, "y")
    dna = build_dna(profile)
    plan = plan_preprocessing(profile, dna, ModelFamily.LINEAR)
    candidate = CandidateModel(
        name="Logistic Regression",
        family=ModelFamily.LINEAR,
        estimator_key="logistic_regression",
        why_included="test",
        explainable=True,
        complexity="low",
    )
    pipe = build_pipeline(plan, candidate, ProblemType.BINARY_CLASSIFICATION, 0)
    pipe.fit(train[["num", "cat"]], train["y"])
    unseen = pd.DataFrame({"num": [None], "cat": ["brand_new"]})
    pred = pipe.predict(unseen)
    assert len(pred) == 1
