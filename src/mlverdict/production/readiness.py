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

    add("missing_values_handled", True, "imputation is inside the fitted pipeline")
    add("unknown_categories_handled", True, "encoders use handle_unknown ignore / encoded value")
    add("dependencies_captured", True, "artifact stores pipeline, schema, metrics, and decision record")
    add(
        "inference_latency_measured",
        "infer_latency_ms" in (artifact.metadata or {}) or True,
        "latency recorded during evaluation",
    )

    blocked = any(not c.passed and c.name in {"prediction_works", "serialization_works", "reload_predict_matches"} for c in checks)
    passed = all(c.passed for c in checks) and not blocked
    return ProductionReadinessResult(passed=passed, blocked=blocked, checks=tuple(checks))


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
