"""Production gate. A critical failure blocks deployment even if a model was selected."""

from __future__ import annotations

import io
from typing import Any

import joblib
import pandas as pd

from mlverdict.core.types import FeatureSchema, ProductionCheck, ProductionReadinessResult
from mlverdict.production.artifact import ModelArtifact


def assess_readiness(
    artifact: ModelArtifact,
    sample: pd.DataFrame,
    schema: FeatureSchema,
) -> ProductionReadinessResult:
    checks: list[ProductionCheck] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(ProductionCheck(name, passed, detail))

    add("model_loads", artifact.pipeline is not None, "sklearn pipeline is present")
    add("schema_available", bool(schema.feature_names), f"{len(schema.feature_names)} features recorded")
    add("required_features_listed", bool(schema.numeric_columns or schema.categorical_columns), "feature roles recorded")

    try:
        preds = artifact.predict(sample)
        add("prediction_works", len(preds) == len(sample), f"predicted {len(preds)} rows")
    except Exception as exc:
        add("prediction_works", False, str(exc))
        preds = None

    try:
        buf = io.BytesIO()
        joblib.dump(artifact.pipeline, buf)
        buf.seek(0)
        reloaded = joblib.load(buf)
        add("serialization_works", reloaded is not None, "joblib round-trip of the pipeline succeeded")
        again = reloaded.predict(sample)
        if preds is not None:
            import numpy as np

            same = bool(np.array_equal(np.asarray(preds), np.asarray(again)))
            add("reload_predict_matches", same, "predictions match after reload")
        else:
            add("reload_predict_matches", False, "original prediction failed")
    except Exception as exc:
        add("serialization_works", False, str(exc))
        add("reload_predict_matches", False, str(exc))

    # Real probes, not assertions: each one exercises the fitted pipeline.
    feature_cols = [c for c in schema.feature_names if c in sample.columns]
    probe = sample.head(min(4, len(sample))).copy()
    try:
        holes = probe.copy()
        for col in feature_cols:
            holes[col] = None
        artifact.predict(holes)
        add("missing_values_handled", True, "an all-missing row was imputed and predicted")
    except Exception as exc:
        add("missing_values_handled", False, f"all-missing row failed: {exc}")

    if schema.categorical_columns:
        try:
            unseen = probe.copy()
            for col in schema.categorical_columns:
                if col in unseen.columns:
                    unseen[col] = "__mlverdict_unseen_category__"
            artifact.predict(unseen)
            add("unknown_categories_handled", True, "an unseen category value was predicted without error")
        except Exception as exc:
            add("unknown_categories_handled", False, f"unseen category failed: {exc}")
    else:
        add("unknown_categories_handled", True, "no categorical features")

    meta = artifact.metadata or {}
    env = meta.get("environment") or {}
    add(
        "dependencies_captured",
        bool(env.get("python")) and bool(env.get("scikit-learn")),
        f"environment recorded: {sorted(env)}" if env else "no environment versions recorded",
    )
    latency = meta.get("infer_latency_ms")
    add(
        "inference_latency_measured",
        isinstance(latency, (int, float)) and latency == latency and latency != float("inf"),
        f"infer_latency_ms={latency}",
    )

    blocked = any(not c.passed and c.name in {"prediction_works", "serialization_works", "reload_predict_matches"} for c in checks)
    passed = all(c.passed for c in checks) and not blocked
    return ProductionReadinessResult(passed=passed, blocked=blocked, checks=tuple(checks))


def environment_versions() -> dict[str, str]:
    """Library versions the artifact was trained with; mismatches are the usual reload failure."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for name in ("mlverdict", "scikit-learn", "numpy", "pandas", "joblib", "xgboost", "lightgbm", "catboost"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def artifact_from_parts(
    pipeline: Any,
    schema: FeatureSchema,
    metadata: dict,
    metrics: dict,
    decision_record: Any,
    configuration: dict,
) -> ModelArtifact:
    return ModelArtifact(
        pipeline=pipeline,
        schema=schema,
        metadata=metadata,
        metrics=metrics,
        decision_record=decision_record,
        configuration=configuration,
    )
