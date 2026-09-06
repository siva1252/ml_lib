from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import ProblemType, ValidationStrategy
from mlverdict.core.types import ProblemDefinition
from mlverdict.data.dna import build_dna
from mlverdict.data.profiler import profile_dataset
from mlverdict.metrics.intelligence import select_metric_plan
from mlverdict.problem.detector import detect_problem
from mlverdict.validation.strategy import select_validation_plan


def _problem(frame, target, override=None):
    profile = profile_dataset(frame, target)
    return profile, detect_problem(profile, frame[target], user_problem_type=override)


def test_imbalanced_classification_uses_stratified(imbalanced_frame):
    profile, problem = _problem(imbalanced_frame, "churn")
    dna = build_dna(profile)
    plan = select_validation_plan(dna, problem, VerdictConfig())
    assert plan.strategy == ValidationStrategy.STRATIFIED_KFOLD
    assert plan.stratify


def test_group_col_is_required_for_grouped_split(grouped_frame):
    profile, problem = _problem(grouped_frame, "churn")
    dna = build_dna(profile)
    hinted = select_validation_plan(dna, problem, VerdictConfig())
    assert hinted.strategy != ValidationStrategy.GROUP_KFOLD
    assert any("group_col" in w for w in hinted.warnings)

    confirmed = select_validation_plan(dna, problem, VerdictConfig(), group_col="customer_id")
    assert confirmed.strategy == ValidationStrategy.GROUP_KFOLD


def test_time_col_selects_time_based(time_frame):
    profile, problem = _problem(time_frame, "churn")
    dna = build_dna(profile, time_col="event_date")
    plan = select_validation_plan(dna, problem, VerdictConfig(), time_col="event_date")
    assert plan.strategy == ValidationStrategy.TIME_BASED
    assert plan.shuffle is False


def test_imbalanced_metric_is_not_accuracy(imbalanced_frame):
    profile, problem = _problem(imbalanced_frame, "churn")
    dna = build_dna(profile)
    plan = select_metric_plan(dna, problem)
    assert plan.primary.name == "pr_auc"
    assert plan.primary.name != "accuracy"


def test_user_metric_is_respected(binary_frame):
    profile, problem = _problem(binary_frame, "churn")
    dna = build_dna(profile)
    plan = select_metric_plan(dna, problem, user_metric="recall")
    assert plan.primary.name == "recall"
    assert plan.user_override


def test_regression_default_metric(regression_frame):
    profile, problem = _problem(regression_frame, "price")
    dna = build_dna(profile)
    plan = select_metric_plan(dna, problem)
    assert plan.primary.name == "rmse"
    assert problem.problem_type == ProblemType.REGRESSION
