from mlverdict import DecisionStatus, Verdict
from mlverdict.api.cli import main


def test_print_run_is_readable_not_a_dict_dump(binary_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, "churn")
    text = str(run)
    assert "MLVerdict" in text
    assert "Leaderboard" in text
    assert "Held-out test" in text
    assert "What to do next" in text
    assert "run.artifact().save" in text
    assert "Selected model" in text
    assert "dtype:" not in text
    assert "DecisionStatus." not in text
    assert "Untouched final-test {" not in text
    assert run.summary().startswith("MLVerdict")
    assert "CV=" in run.summary()
    str(run).encode("cp1252")
    run.report().encode("cp1252")


def test_display_prints_the_verdict(binary_frame, fast_config, capsys):
    run = Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, "churn")
    returned = run.display()
    out = capsys.readouterr().out
    assert returned == str(run)
    assert out.strip() == str(run).strip()


def test_report_formats_metrics_not_raw_dict(binary_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(binary_frame, "churn")
    report = run.report()
    assert "Executive summary" in report
    assert "Held-out test" in report or "Final untouched test" in report
    assert "What to do next" in report
    assert "Untouched final-test {" not in report
    assert "Attention" in report


def test_imbalanced_zero_catch_surfaces_in_console_when_present(imbalanced_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(imbalanced_frame, "churn")
    text = str(run)
    assert "pr_auc" in text.lower() or (run.metric_plan and run.metric_plan.primary.name == "pr_auc")
    if run.final_test and run.final_test.metrics.get("f1", 1) == 0:
        assert "ATTENTION" in text
        assert "0.5 cutoff" in text or "positive class" in text


def test_undecided_console_tells_user_what_to_pass(ambiguous_target_frame, fast_config):
    run = Verdict(config=fast_config, enable_hpo=False).fit(ambiguous_target_frame, "score")
    assert run.status == DecisionStatus.UNDECIDED
    text = str(run)
    assert "problem_type" in text
    assert "run.artifact().save" not in text


def test_cli_fit_fast_prints_verdict_and_saves(binary_frame, tmp_path, capsys):
    csv = tmp_path / "data.csv"
    binary_frame.to_csv(csv, index=False)
    artifact = tmp_path / "model.joblib"
    code = main(
        [
            "fit",
            str(csv),
            "--target",
            "churn",
            "--fast",
            "--no-hpo",
            "--save",
            str(artifact),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "MLVerdict" in captured.out
    assert "Leaderboard" in captured.out
    assert artifact.exists()
    assert "Saved artifact" in captured.out


def test_cli_missing_file_is_nonzero():
    code = main(["fit", "no_such_file_mlverdict.csv", "--target", "y", "--fast"])
    assert code == 2
