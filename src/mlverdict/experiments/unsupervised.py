"""Unsupervised CV and HPO. Scores structure on features; never invents a label."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from mlverdict.core.config import VerdictConfig
from mlverdict.core.enums import ExperimentStatus, ModelFamily, ProblemType
from mlverdict.core.types import (
    CandidateModel,
    DatasetDNA,
    DatasetProfile,
    ExperimentResult,
    MetricPlan,
    ProblemDefinition,
    ValidationPlan,
)
from mlverdict.experiments.cv import _latency_ms
from mlverdict.experiments.hpo import (
    best_params_or_default,
    has_search_space,
    np_finite,
    select_promising,
)
from mlverdict.experiments.scoring import extract_score
from mlverdict.preprocessing.pipeline import build_pipeline
from mlverdict.preprocessing.planner import plan_preprocessing
from mlverdict.validation.splitter import build_cv_splitter

optuna.logging.set_verbosity(optuna.logging.WARNING)


def policy_params(estimator_key: str, config: VerdictConfig) -> dict[str, Any]:
    """Settings fixed by policy, not tuned: every detector flags the same expected share,
    and PCA keeps the fewest components reaching the variance threshold.

    Tuning these against an unlabeled score is circular (a detector "wins" by flagging more rows,
    PCA "wins" by keeping every component), so they are set once from the config.
    """
    if estimator_key in {"isolation_forest", "local_outlier_factor"}:
        return {"contamination": config.anomaly_contamination}
    if estimator_key == "one_class_svm":
        return {"nu": config.anomaly_contamination}
    if estimator_key == "detector_ensemble":
        return {"contamination": config.anomaly_contamination, "max_rows_for_svm": config.max_rows_for_svm}
    if estimator_key == "pca":
        return {"n_components": config.pca_variance_threshold}
    return {}


def compute_unsupervised_metrics(
    pipeline,
    X: pd.DataFrame,
    problem: ProblemDefinition,
    *,
    silhouette_sample_size: int = 5_000,
    random_state: int = 0,
) -> dict[str, float]:
    """Internal indices / reconstruction. No labeled accuracy."""
    if problem.problem_type is None or len(X) == 0:
        return {}
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    xt = np.asarray(preprocess.transform(X), dtype=float)
    out: dict[str, float] = {}
    kind = problem.problem_type

    if kind == ProblemType.CLUSTERING:
        labels = np.asarray(pipeline.predict(X))
        n_labels = int(len(np.unique(labels)))
        out["n_clusters"] = float(n_labels)
        if n_labels >= 2 and n_labels < len(labels) and len(labels) >= 4:
            # silhouette is O(n^2) in memory and time; a fixed-seed sample keeps large data tractable.
            sample = silhouette_sample_size if len(labels) > silhouette_sample_size else None
            out["silhouette"] = _safe(
                lambda: float(silhouette_score(xt, labels, sample_size=sample, random_state=random_state))
            )
            out["calinski_harabasz"] = _safe(lambda: float(calinski_harabasz_score(xt, labels)))
            out["davies_bouldin"] = _safe(lambda: float(davies_bouldin_score(xt, labels)))
        return {k: v for k, v in out.items() if v == v}

    if kind == ProblemType.ANOMALY_DETECTION:
        pred = np.asarray(pipeline.predict(X))
        flagged = pred == -1
        out["outlier_rate"] = float(np.mean(flagged))
        if hasattr(model, "score_samples"):
            normality = np.asarray(model.score_samples(xt), dtype=float).ravel()
            spread = float(np.std(normality))
            if 0 < flagged.sum() < len(flagged) and spread > 0:
                # How far the flagged rows sit from the rest, in units of score spread.
                # Every detector flags the same share (policy), so this compares how cleanly
                # each one isolates its outliers rather than how many it flags.
                gap = float(normality[~flagged].mean() - normality[flagged].mean())
                out["anomaly_separation"] = gap / spread
            else:
                out["anomaly_separation"] = 0.0
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(xt), dtype=float).ravel()
            out["decision_std"] = float(np.std(scores))
            out["decision_mean"] = float(np.mean(scores))
        return {k: v for k, v in out.items() if v == v}

    if kind == ProblemType.DIMENSIONALITY_REDUCTION:
        z = np.asarray(pipeline.predict(X), dtype=float)
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        out["n_components"] = float(z.shape[1])
        ratio = getattr(model, "explained_variance_ratio_", None)
        if ratio is not None:
            out["explained_variance"] = float(np.sum(ratio))
        if hasattr(model, "inverse_transform"):
            reconstructed = np.asarray(model.inverse_transform(z), dtype=float)
            out["reconstruction_rmse"] = float(np.sqrt(np.mean((xt - reconstructed) ** 2)))
        return {k: v for k, v in out.items() if v == v}

    return {}


def _safe(fn) -> float:
    try:
        return float(fn())
    except Exception:
        return float("nan")


def run_unsupervised_cv(
    pipeline,
    X: pd.DataFrame,
    candidate: CandidateModel,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
    validation: ValidationPlan,
    *,
    random_state: int,
    groups: pd.Series | None = None,
    latency_probe_rows: int = 64,
    configuration: dict[str, Any] | None = None,
    optimized: bool = False,
    silhouette_sample_size: int = 5_000,
) -> ExperimentResult:
    n_groups = int(groups.nunique()) if groups is not None else None
    splitter = build_cv_splitter(validation, random_state, len(X), n_groups)
    fold_scores: list[float] = []
    train_scores: list[float] = []
    metric_accum: dict[str, list[float]] = {}
    last_model = None
    last_val_x = X
    t0 = time.perf_counter()
    error = None
    status = ExperimentStatus.SUCCESS

    try:
        X_arr = X.reset_index(drop=True)
        dummy = np.zeros(len(X_arr))
        g_arr = groups.reset_index(drop=True) if groups is not None else None
        if g_arr is not None:
            splits = splitter.split(X_arr, dummy, g_arr)
        else:
            splits = splitter.split(X_arr, dummy)

        for train_idx, val_idx in splits:
            model = clone(pipeline)
            X_tr, X_va = X_arr.iloc[train_idx], X_arr.iloc[val_idx]
            model.fit(X_tr)
            metrics_va = compute_unsupervised_metrics(
                model, X_va, problem, silhouette_sample_size=silhouette_sample_size, random_state=random_state
            )
            fold_scores.append(extract_score(metrics_va, metric_plan))
            for key, value in metrics_va.items():
                metric_accum.setdefault(key, []).append(value)
            metrics_tr = compute_unsupervised_metrics(
                model, X_tr, problem, silhouette_sample_size=silhouette_sample_size, random_state=random_state
            )
            train_scores.append(extract_score(metrics_tr, metric_plan))
            last_model = model
            last_val_x = X_va
    except Exception as exc:
        status = ExperimentStatus.FAILED
        error = str(exc)
        fold_scores = [float("nan")]

    train_time = time.perf_counter() - t0
    scores = np.asarray(fold_scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    if status == ExperimentStatus.SUCCESS and len(finite) == 0:
        status = ExperimentStatus.FAILED
        error = error or f"No finite '{metric_plan.primary.name}' scores on validation folds."
    mean = float(finite.mean()) if len(finite) else float("nan")
    std = float(finite.std(ddof=0)) if len(finite) else float("nan")
    mn = float(finite.min()) if len(finite) else float("nan")
    mx = float(finite.max()) if len(finite) else float("nan")
    train_mean = float(np.nanmean(train_scores)) if train_scores else None
    gap = None
    if train_mean is not None and np.isfinite(mean) and np.isfinite(train_mean):
        gap = float(train_mean - mean)
    latency = _latency_ms(last_model, last_val_x, latency_probe_rows) if last_model is not None else float("inf")
    agg_metrics = {k: float(np.nanmean(v)) for k, v in metric_accum.items()}
    if metric_plan.primary.name not in agg_metrics and np.isfinite(mean):
        raw = mean if metric_plan.primary.greater_is_better else -mean
        agg_metrics[metric_plan.primary.name] = float(raw)

    return ExperimentResult(
        model_name=candidate.name,
        estimator_key=candidate.estimator_key,
        configuration=configuration or dict(candidate.params),
        metrics=agg_metrics,
        fold_scores=tuple(float(s) for s in fold_scores),
        mean_score=mean,
        std_score=std if np.isfinite(std) else 0.0,
        min_score=mn,
        max_score=mx,
        train_score=train_mean,
        generalization_gap=gap,
        train_time_seconds=float(train_time),
        infer_latency_ms=float(latency),
        status=status,
        error=error,
        n_samples=len(X),
        n_features=X.shape[1],
        validation_strategy=validation.strategy.value,
        optimized=optimized,
        family=candidate.family.value if hasattr(candidate.family, "value") else str(candidate.family),
        explainable=candidate.explainable,
        complexity=candidate.complexity,
    )


def _space(trial: optuna.Trial, key: str, max_clusters: int = 10) -> dict[str, Any]:
    """Structural knobs only. Outlier share and PCA size are policy (see ``policy_params``)."""
    if key in {"kmeans", "gmm", "birch"}:
        return {"n_clusters": trial.suggest_int("n_clusters", 2, max_clusters)}
    if key == "isolation_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 200)}
    if key == "one_class_svm":
        return {"gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True)}
    if key == "local_outlier_factor":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 8, 50)}
    return {}


def optimize_unsupervised_candidate(
    candidate: CandidateModel,
    X: pd.DataFrame,
    profile: DatasetProfile,
    dna: DatasetDNA,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
    validation: ValidationPlan,
    config: VerdictConfig,
    *,
    groups: pd.Series | None = None,
    extra_drop: tuple[str, ...] = (),
) -> ExperimentResult:
    assert problem.problem_type is not None
    family = candidate.family if isinstance(candidate.family, ModelFamily) else ModelFamily(candidate.family)
    plan = plan_preprocessing(profile, dna, family, config, extra_drop=extra_drop)

    policy = policy_params(candidate.estimator_key, config)

    def space(trial: optuna.Trial, key: str) -> dict[str, Any]:
        return _space(trial, key, config.max_clusters)

    if not has_search_space(space, candidate.estimator_key):
        pipe = build_pipeline(plan, candidate, problem.problem_type, config.random_state, dict(policy))
        return run_unsupervised_cv(
            pipe,
            X,
            candidate,
            problem,
            metric_plan,
            validation,
            random_state=config.random_state,
            groups=groups,
            latency_probe_rows=config.latency_probe_rows,
            configuration=dict(policy),
            optimized=False,
            silhouette_sample_size=config.silhouette_sample_size,
        )

    def objective(trial: optuna.Trial) -> float:
        params = {**policy, **space(trial, candidate.estimator_key)}
        pipe = build_pipeline(plan, candidate, problem.problem_type, config.random_state, params)
        result = run_unsupervised_cv(
            pipe,
            X,
            candidate,
            problem,
            metric_plan,
            validation,
            random_state=config.random_state,
            groups=groups,
            latency_probe_rows=config.latency_probe_rows,
            configuration=params,
            optimized=True,
            silhouette_sample_size=config.silhouette_sample_size,
        )
        if result.status != ExperimentStatus.SUCCESS or not np_finite(result.mean_score):
            raise optuna.TrialPruned()
        return result.mean_score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
    )
    study.optimize(
        objective,
        n_trials=config.max_hpo_trials,
        timeout=config.max_hpo_time_seconds,
        catch=(Exception,),
    )
    best_params = {**policy, **best_params_or_default(study)}
    pipe = build_pipeline(plan, candidate, problem.problem_type, config.random_state, best_params)
    return run_unsupervised_cv(
        pipe,
        X,
        candidate,
        problem,
        metric_plan,
        validation,
        random_state=config.random_state,
        groups=groups,
        latency_probe_rows=config.latency_probe_rows,
        configuration=best_params,
        optimized=True,
        silhouette_sample_size=config.silhouette_sample_size,
    )


def _scan_clusters(
    candidate: CandidateModel,
    X: pd.DataFrame,
    profile: DatasetProfile,
    dna: DatasetDNA,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
    validation: ValidationPlan,
    config: VerdictConfig,
    *,
    groups: pd.Series | None,
    extra_drop: tuple[str, ...],
) -> ExperimentResult:
    """Try every k in 2..max_clusters; a handful of random trials can miss the right k.

    On large data k is chosen on a fixed-seed sample of ``cluster_scan_rows`` rows, then the
    chosen k is cross-validated on every row so reported scores describe the full data.
    """
    family = candidate.family if isinstance(candidate.family, ModelFamily) else ModelFamily(candidate.family)
    plan = plan_preprocessing(profile, dna, family, config, extra_drop=extra_drop)
    upper = max(2, min(config.max_clusters, len(X) // 10))
    sampled = len(X) > config.cluster_scan_rows
    if sampled:
        rows = X.sample(n=config.cluster_scan_rows, random_state=config.random_state).index
        X_scan = X.loc[rows]
        groups_scan = groups.loc[rows] if groups is not None else None
    else:
        X_scan, groups_scan = X, groups
    results = []
    for k in range(2, upper + 1):
        params = {"n_clusters": k}
        pipe = build_pipeline(plan, candidate, problem.problem_type, config.random_state, params)
        results.append(
            run_unsupervised_cv(
                pipe,
                X_scan,
                candidate,
                problem,
                metric_plan,
                validation,
                random_state=config.random_state,
                latency_probe_rows=config.latency_probe_rows,
                configuration=params,
                optimized=True,
                silhouette_sample_size=config.silhouette_sample_size,
                groups=groups_scan,
            )
        )
    ok = [r for r in results if r.status == ExperimentStatus.SUCCESS and np_finite(r.mean_score)]
    if not ok:
        return results[0]
    best = max(ok, key=lambda r: r.mean_score)
    if sampled:
        params = dict(best.configuration)
        pipe = build_pipeline(plan, candidate, problem.problem_type, config.random_state, params)
        best = run_unsupervised_cv(
            pipe,
            X,
            candidate,
            problem,
            metric_plan,
            validation,
            random_state=config.random_state,
            groups=groups,
            latency_probe_rows=config.latency_probe_rows,
            configuration=params,
            optimized=True,
            silhouette_sample_size=config.silhouette_sample_size,
        )
    metrics = dict(best.metrics)
    metrics["k_scanned"] = float(len(results))
    return replace(best, metrics=metrics)


def run_unsupervised_experiments(
    X: pd.DataFrame,
    candidates: tuple[CandidateModel, ...],
    profile: DatasetProfile,
    dna: DatasetDNA,
    problem: ProblemDefinition,
    metric_plan: MetricPlan,
    validation: ValidationPlan,
    config: VerdictConfig,
    *,
    groups: pd.Series | None = None,
    extra_drop: tuple[str, ...] = (),
    enable_hpo: bool = True,
) -> tuple[ExperimentResult, ...]:
    results: list[ExperimentResult] = []
    assert problem.problem_type is not None
    if problem.problem_type == ProblemType.CLUSTERING:
        return tuple(
            _scan_clusters(
                c, X, profile, dna, problem, metric_plan, validation, config, groups=groups, extra_drop=extra_drop
            )
            for c in candidates
        )
    anomaly = problem.problem_type == ProblemType.ANOMALY_DETECTION
    for candidate in candidates:
        family = candidate.family if isinstance(candidate.family, ModelFamily) else ModelFamily(candidate.family)
        plan = plan_preprocessing(profile, dna, family, config, extra_drop=extra_drop)
        policy = policy_params(candidate.estimator_key, config)
        pipeline = build_pipeline(plan, candidate, problem.problem_type, config.random_state, dict(policy))
        results.append(
            run_unsupervised_cv(
                pipeline,
                X,
                candidate,
                problem,
                metric_plan,
                validation,
                random_state=config.random_state,
                groups=groups,
                latency_probe_rows=config.latency_probe_rows,
                configuration=dict(policy),
                silhouette_sample_size=config.silhouette_sample_size,
            )
        )
    baselines = tuple(results)
    if anomaly:
        # Detector settings are policy; tuning against an unlabeled self-score is circular.
        return baselines
    if not enable_hpo or config.max_hpo_trials <= 0:
        return baselines

    by_name = {c.name: c for c in candidates}
    tunable = tuple(
        b
        for b in baselines
        if has_search_space(lambda t, k: _space(t, k, config.max_clusters), b.estimator_key)
    )
    promising = select_promising(tunable, config.n_hpo_candidates)
    optimized: list[ExperimentResult] = []
    optimized_names: set[str] = set()
    for base in promising:
        candidate = by_name.get(base.model_name)
        if candidate is None:
            continue
        result = optimize_unsupervised_candidate(
            candidate,
            X,
            profile,
            dna,
            problem,
            metric_plan,
            validation,
            config,
            groups=groups,
            extra_drop=extra_drop,
        )
        optimized.append(result)
        optimized_names.add(result.model_name)
    merged = [r for r in baselines if r.model_name not in optimized_names]
    merged.extend(optimized)
    return tuple(merged)
