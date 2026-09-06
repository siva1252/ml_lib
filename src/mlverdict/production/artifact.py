"""Deployable artifact: model + preprocess + schema + metrics + decision record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from mlverdict.core.exceptions import ArtifactError
from mlverdict.core.types import FeatureSchema, ModelDecisionRecord


@dataclass
class ModelArtifact:
    pipeline: Any
    schema: FeatureSchema
    metadata: dict[str, Any]
    metrics: dict[str, float]
    decision_record: ModelDecisionRecord | dict[str, Any]
    configuration: dict[str, Any]

    def predict(self, data: pd.DataFrame | list[dict[str, Any]]) -> Any:
        frame = _align_frame(data, self.schema)
        try:
            return self.pipeline.predict(frame)
        except Exception as exc:
            raise ArtifactError(f"Prediction failed: {exc}") from exc

    def predict_proba(self, data: pd.DataFrame | list[dict[str, Any]]) -> Any:
        frame = _align_frame(data, self.schema)
        if not hasattr(self.pipeline, "predict_proba"):
            raise ArtifactError("This artifact does not support predict_proba.")
        return self.pipeline.predict_proba(frame)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "pipeline": self.pipeline,
            "schema": self.schema,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "decision_record": self.decision_record,
            "configuration": self.configuration,
        }
        try:
            joblib.dump(payload, path)
        except Exception as exc:
            raise ArtifactError(f"Failed to save artifact: {exc}") from exc
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ModelArtifact":
        path = Path(path)
        if not path.exists():
            raise ArtifactError(f"Artifact not found: {path}")
        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ArtifactError(f"Failed to load artifact: {exc}") from exc
        try:
            return cls(
                pipeline=payload["pipeline"],
                schema=payload["schema"],
                metadata=payload.get("metadata", {}),
                metrics=payload.get("metrics", {}),
                decision_record=payload.get("decision_record", {}),
                configuration=payload.get("configuration", {}),
            )
        except KeyError as exc:
            raise ArtifactError(f"Artifact is missing required key: {exc}") from exc


def _align_frame(data: pd.DataFrame | list[dict[str, Any]], schema: FeatureSchema) -> pd.DataFrame:
    if isinstance(data, list):
        frame = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        raise ArtifactError("Predict input must be a DataFrame or a list of records.")
    missing = [c for c in schema.feature_names if c not in frame.columns]
    if missing:
        # allow dropped identifier/constant columns to be absent
        required = [
            c
            for c in schema.feature_names
            if c in schema.numeric_columns + schema.categorical_columns + schema.datetime_columns
        ]
        missing_required = [c for c in required if c not in frame.columns]
        if missing_required:
            raise ArtifactError(f"Missing required features: {missing_required}")
        for col in missing:
            frame[col] = None
    return frame.reindex(columns=list(schema.feature_names))
