"""Build a single picklable preprocess+model sklearn pipeline. Fit only on train data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from mlverdict.core.enums import ProblemType
from mlverdict.core.types import CandidateModel, PreprocessPlan
from mlverdict.models.adapters import build_estimator


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """Impute, encode, scale, and expand datetimes. Unknown categories are handled."""

    def __init__(
        self,
        numeric_columns: tuple[str, ...] = (),
        categorical_columns: tuple[str, ...] = (),
        datetime_columns: tuple[str, ...] = (),
        numeric_scaling: bool = True,
        categorical_encoding: str = "ordinal",
    ) -> None:
        self.numeric_columns = tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.datetime_columns = tuple(datetime_columns)
        self.numeric_scaling = numeric_scaling
        self.categorical_encoding = categorical_encoding

    def _expand(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for col in self.datetime_columns:
            if col not in out.columns:
                continue
            dt = pd.to_datetime(out[col], errors="coerce")
            out[f"{col}_year"] = dt.dt.year.astype(float)
            out[f"{col}_month"] = dt.dt.month.astype(float)
            out[f"{col}_day"] = dt.dt.day.astype(float)
            out[f"{col}_dow"] = dt.dt.dayofweek.astype(float)
            out = out.drop(columns=[col])
        return out

    def _numeric_names(self) -> list[str]:
        names = list(self.numeric_columns)
        for col in self.datetime_columns:
            names.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dow"])
        return names

    def fit(self, X, y=None):
        frame = self._expand(_as_frame(X))
        self.numeric_in_ = [c for c in self._numeric_names() if c in frame.columns]
        self.categorical_in_ = [c for c in self.categorical_columns if c in frame.columns]
        if self.numeric_in_:
            num = frame[self.numeric_in_].apply(pd.to_numeric, errors="coerce")
            self.num_imputer_ = SimpleImputer(strategy="median").fit(num)
            self.scaler_ = StandardScaler().fit(self.num_imputer_.transform(num)) if self.numeric_scaling else None
        else:
            self.num_imputer_ = None
            self.scaler_ = None
        if self.categorical_in_:
            cats = frame[self.categorical_in_].astype("object")
            self.cat_imputer_ = SimpleImputer(strategy="most_frequent").fit(cats)
            imputed = self.cat_imputer_.transform(cats)
            if self.categorical_encoding == "onehot_or_ordinal":
                card = [len(pd.unique(imputed[:, i])) for i in range(imputed.shape[1])]
                self.onehot_idx_ = [i for i, c in enumerate(card) if c <= 20]
                self.ordinal_idx_ = [i for i, c in enumerate(card) if c > 20]
                self.onehot_ = (
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(imputed[:, self.onehot_idx_])
                    if self.onehot_idx_
                    else None
                )
                self.ordinal_ = (
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
                        imputed[:, self.ordinal_idx_]
                    )
                    if self.ordinal_idx_
                    else None
                )
            else:
                self.onehot_ = None
                self.onehot_idx_ = []
                self.ordinal_idx_ = list(range(imputed.shape[1]))
                self.ordinal_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(imputed)
        else:
            self.cat_imputer_ = None
            self.onehot_ = None
            self.ordinal_ = None
            self.onehot_idx_ = []
            self.ordinal_idx_ = []
        return self

    def transform(self, X):
        frame = self._expand(_as_frame(X))
        parts: list[np.ndarray] = []
        if self.numeric_in_:
            num = frame.reindex(columns=self.numeric_in_)
            num = num.apply(pd.to_numeric, errors="coerce")
            arr = self.num_imputer_.transform(num)
            if self.scaler_ is not None:
                arr = self.scaler_.transform(arr)
            parts.append(arr)
        if self.categorical_in_:
            cats = frame.reindex(columns=self.categorical_in_).astype("object")
            imputed = self.cat_imputer_.transform(cats)
            if self.onehot_ is not None and self.onehot_idx_:
                parts.append(self.onehot_.transform(imputed[:, self.onehot_idx_]))
            if self.ordinal_ is not None and self.ordinal_idx_:
                parts.append(self.ordinal_.transform(imputed[:, self.ordinal_idx_]))
        if not parts:
            return np.zeros((len(frame), 1), dtype=float)
        return np.hstack(parts)


def build_pipeline(
    plan: PreprocessPlan,
    candidate: CandidateModel,
    problem_type: ProblemType,
    random_state: int,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    preprocess = TabularPreprocessor(
        numeric_columns=plan.numeric_columns,
        categorical_columns=plan.categorical_columns,
        datetime_columns=plan.datetime_columns,
        numeric_scaling=plan.numeric_scaling,
        categorical_encoding=plan.categorical_encoding,
    )
    model = build_estimator(candidate.estimator_key, problem_type, random_state, params)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def clone_pipeline(pipeline: Pipeline) -> Pipeline:
    return clone(pipeline)


def is_classifier(estimator) -> bool:
    step = estimator.named_steps["model"] if isinstance(estimator, Pipeline) else estimator
    return isinstance(step, ClassifierMixin) or hasattr(step, "predict_proba") and not isinstance(step, RegressorMixin)
