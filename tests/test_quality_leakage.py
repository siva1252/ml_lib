from mlverdict.core.enums import Severity
from mlverdict.data.dna import build_dna
from mlverdict.data.profiler import profile_dataset
from mlverdict.quality.analyzer import analyze_quality
from mlverdict.quality.leakage import detect_leakage


def test_quality_flags_missing_constant_id_and_duplicates(messy_frame):
    profile = profile_dataset(messy_frame, "churn")
    report = analyze_quality(messy_frame, profile)
    types = {i.issue_type for i in report.issues}
    assert "missing_values" in types
    assert "constant_feature" in types
    assert "potential_identifier" in types
    assert "duplicate_rows" in types
    assert all(i.recommendation for i in report.issues)


def test_leakage_signals_do_not_claim_clean(leakage_frame):
    profile = profile_dataset(leakage_frame, "churn")
    dna = build_dna(profile)
    report = detect_leakage(leakage_frame, profile, dna, "churn")
    kinds = {s.signal_type for s in report.signals}
    assert "target_derived_name" in kinds or "perfect_target_correlation" in kinds
    assert "post_outcome_or_future" in kinds
    assert "id_leakage" in kinds
    assert "not a guarantee" in report.claim.lower() or "signals" in report.claim.lower()
    assert any(s.severity == Severity.HIGH_RISK for s in report.signals)
