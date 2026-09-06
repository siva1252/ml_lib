from mlverdict.core.enums import DecisionStatus, ProblemType
from mlverdict.data.profiler import profile_dataset
from mlverdict.problem.detector import detect_problem


def test_binary_detection(binary_frame):
    profile = profile_dataset(binary_frame, "churn")
    problem = detect_problem(profile, binary_frame["churn"])
    assert problem.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert problem.status == DecisionStatus.DECIDED
    assert problem.confidence.value == "high"


def test_multiclass_detection(multiclass_frame):
    profile = profile_dataset(multiclass_frame, "label")
    problem = detect_problem(profile, multiclass_frame["label"])
    assert problem.problem_type == ProblemType.MULTICLASS_CLASSIFICATION
    assert problem.status == DecisionStatus.DECIDED


def test_regression_detection(regression_frame):
    profile = profile_dataset(regression_frame, "price")
    problem = detect_problem(profile, regression_frame["price"])
    assert problem.problem_type == ProblemType.REGRESSION
    assert problem.status == DecisionStatus.DECIDED


def test_ambiguous_target_is_undecided(ambiguous_target_frame):
    profile = profile_dataset(ambiguous_target_frame, "score")
    problem = detect_problem(profile, ambiguous_target_frame["score"])
    assert problem.status == DecisionStatus.UNDECIDED
    assert problem.problem_type is None
    assert problem.confidence.value == "low"


def test_user_override_wins(ambiguous_target_frame):
    profile = profile_dataset(ambiguous_target_frame, "score")
    problem = detect_problem(
        profile,
        ambiguous_target_frame["score"],
        user_problem_type="regression",
    )
    assert problem.user_override
    assert problem.problem_type == ProblemType.REGRESSION
    assert problem.status == DecisionStatus.DECIDED
