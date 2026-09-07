from mlverdict.core.enums import DecisionStatus, Severity
from mlverdict.core.types import LeakageReport, LeakageSignal, QualityIssue, QualityReport
from mlverdict.reporting.insights import (
    interpret_final_test,
    interpret_gap,
    metric_help,
    next_steps,
    watchlist,
)
from mlverdict.reporting.tables import ascii_table


def test_pr_auc_help_mentions_imbalance():
    text = metric_help("pr_auc")
    assert "imbalance" in text.lower() or "rare" in text.lower()


def test_majority_class_trap_from_user_report_numbers():
    metrics = {
        "accuracy": 0.8,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "balanced_accuracy": 0.5,
        "roc_auc": 0.84375,
        "pr_auc": 0.6467276975897666,
        "log_loss": 0.4427198063321553,
    }
    notes = interpret_final_test(metrics, is_classification=True, n_rows=80)
    blob = " ".join(notes).lower()
    assert "positive class" in blob or "f1" in blob
    assert "majority" in blob or "0.5" in blob or "cutoff" in blob
    assert "80" in " ".join(notes)


def test_regression_skips_class_trap():
    notes = interpret_final_test({"rmse": 1.2, "mae": 0.8}, is_classification=False, n_rows=500)
    assert notes == ()


def test_gap_explanations():
    assert "overfitting" in (interpret_gap(0.2, 80) or "").lower()
    assert "noise" in (interpret_gap(-0.22, 80) or "").lower()
    assert "ballpark" in (interpret_gap(0.01, 200) or "").lower()


def test_watchlist_filters_info():
    quality = QualityReport(
        issues=(
            QualityIssue("potential_identifier", Severity.HIGH_RISK, "customer_id", "uniq=1", "looks like an id", "drop"),
            QualityIssue("ok", Severity.INFO, "x", "fine", "fine", "none"),
        )
    )
    leakage = LeakageReport(
        signals=(
            LeakageSignal("id_leakage", Severity.HIGH_RISK, "customer_id", "id", "may leak identity", "drop"),
            LeakageSignal("preprocessing_leakage_risk", Severity.INFO, None, "-", "info only", "cv"),
        )
    )
    rows = watchlist(quality, leakage)
    assert len(rows) == 2
    assert all("customer_id" in r for r in rows)


def test_next_steps_decided_vs_blocked():
    ok = next_steps(status=DecisionStatus.DECIDED, has_artifact=True)
    assert any("save" in s for s in ok)
    assert any("serve" in s for s in ok)
    blocked = next_steps(status=DecisionStatus.BLOCKED, has_artifact=False)
    assert any("blocker" in s.lower() or "no deployable" in s.lower() for s in blocked)


def test_ascii_table_alignment():
    text = ascii_table(("Model", "Score"), [("Extra Trees", "0.43"), ("Ridge", "0.10")], aligns=("l", "r"))
    assert "Extra Trees" in text
    assert text.splitlines()[0].startswith("+")
