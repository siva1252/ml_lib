from mlverdict.core.config import Constraints, VerdictConfig
from mlverdict.core.enums import DecisionStatus, ExperimentStatus, ProblemType
from mlverdict.core.types import EvaluationResult, MetricSpec
from mlverdict.data.dna import build_dna
from mlverdict.data.profiler import profile_dataset
from mlverdict.decision.selector import select_model
from mlverdict.metrics.intelligence import select_metric_plan
from mlverdict.models.candidates import select_candidates
from mlverdict.problem.detector import detect_problem


def _eval(name, primary, latency, stability=0.8, gen=0.7, complexity="medium", explainable=True, fails=()):
    return EvaluationResult(
        model_name=name,
        predictive={"pr_auc": primary},
        primary_score=primary,
        stability_score=stability,
        generalization_score=gen,
        latency_ms=latency,
        train_time_seconds=1.0,
        complexity=complexity,
        explainable=explainable,
        constraint_failures=fails,
        passes_constraints=not fails,
        composite_score=0.0,
        fold_scores=(primary, primary),
    )


def test_candidates_are_not_the_full_catalog(binary_frame):
    profile = profile_dataset(binary_frame, "churn")
    dna = build_dna(profile)
    problem = detect_problem(profile, binary_frame["churn"])
    metrics = select_metric_plan(dna, problem)
    chosen = select_candidates(dna, problem, metrics)
    names = {c.name for c in chosen.included}
    assert names
    assert "Logistic Regression" in names or "Random Forest" in names
    # Tiny/small data should not blindly include every booster.
    assert len(chosen.included) < 8


def test_decision_respects_latency_constraint():
    from mlverdict.core.types import MetricPlan

    plan = MetricPlan(
        primary=MetricSpec("pr_auc", True, True, "average_precision"),
        secondary=(),
        evidence=(),
    )
    cat = _eval("CatBoost", 0.90, 180.0, fails=("latency 180.00ms exceeds max_latency_ms=100",))
    rf = _eval("Random Forest", 0.87, 30.0)
    # composite unused by selector except for ranking eligible models
    cat = EvaluationResult(**{**cat.__dict__, "composite_score": 0.95, "passes_constraints": False})
    rf = EvaluationResult(**{**rf.__dict__, "composite_score": 0.70, "passes_constraints": True})
    decision = select_model((cat, rf), plan)
    assert decision.status == DecisionStatus.DECIDED
    assert decision.selected_model == "Random Forest"
    assert any(r.model_name == "CatBoost" for r in decision.rejected)


def test_decision_prefers_stable_model_when_scores_are_close():
    from mlverdict.core.types import MetricPlan

    plan = MetricPlan(primary=MetricSpec("f1", True), secondary=(), evidence=())
    shaky = EvaluationResult(
        **{
            **_eval("Shaky Boost", 0.91, 40.0, stability=0.2).__dict__,
            "composite_score": 0.55,
            "passes_constraints": True,
        }
    )
    stable = EvaluationResult(
        **{
            **_eval("Stable Forest", 0.88, 35.0, stability=0.95).__dict__,
            "composite_score": 0.80,
            "passes_constraints": True,
        }
    )
    decision = select_model((shaky, stable), plan)
    assert decision.selected_model == "Stable Forest"


def test_all_constraint_failures_block():
    from mlverdict.core.types import MetricPlan

    plan = MetricPlan(primary=MetricSpec("f1", True), secondary=(), evidence=())
    only = EvaluationResult(
        **{
            **_eval("Slow", 0.99, 500.0, fails=("latency too high",)).__dict__,
            "passes_constraints": False,
        }
    )
    decision = select_model((only,), plan)
    assert decision.status == DecisionStatus.BLOCKED
    assert decision.selected_model is None


def test_require_explainable_filters_boosting(binary_frame):
    profile = profile_dataset(binary_frame, "churn")
    dna = build_dna(profile)
    problem = detect_problem(profile, binary_frame["churn"])
    metrics = select_metric_plan(dna, problem)
    chosen = select_candidates(dna, problem, metrics, Constraints(require_explainable=True))
    assert all(c.explainable for c in chosen.included)
    assert problem.problem_type == ProblemType.BINARY_CLASSIFICATION
    assert VerdictConfig()
    assert ExperimentStatus.SUCCESS
