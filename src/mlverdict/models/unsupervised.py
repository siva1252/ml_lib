"""Unsupervised estimators. predict() must work on new rows (no agglomerative)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from mlverdict.core.enums import ModelFamily, ProblemType

UNSUPERVISED_TYPES = {
    ProblemType.CLUSTERING,
    ProblemType.ANOMALY_DETECTION,
    ProblemType.DIMENSIONALITY_REDUCTION,
}

UNSUPERVISED_CATALOG = (
    {
        "name": "KMeans",
        "key": "kmeans",
        "family": ModelFamily.UNSUPERVISED,
        "tasks": ("clustering",),
        "explainable": True,
        "complexity": "low",
        "optional": False,
    },
    {
        "name": "Gaussian Mixture",
        "key": "gmm",
        "family": ModelFamily.UNSUPERVISED,
        "tasks": ("clustering",),
        "explainable": False,
        "complexity": "medium",
        "optional": False,
    },
    {
        "name": "Birch",
        "key": "birch",
        "family": ModelFamily.UNSUPERVISED,
        "tasks": ("clustering",),
        "explainable": True,
        "complexity": "low",
        "optional": False,
    },
    {
        "name": "Isolation Forest",
        "key": "isolation_forest",
        "family": ModelFamily.UNSUPERVISED,
        "tasks": ("anomaly_detection",),
        "explainable": True,
        "complexity": "medium",
        "optional": False,
    },
    {
        "name": "One-Class SVM",
        "key": "one_class_svm",
        "family": ModelFamily.UNSUPERVISED,
        "tasks": ("anomaly_detection",),
        "explainable": False,
        "complexity": "medium",
        "optional": False,
    },
    {
        "name": "Local Outlier Factor",
        "key": "local_outlier_factor",
        "family": ModelFamily.UNSUPERVISED,
        "tasks": ("anomaly_detection",),
        "explainable": False,
        "complexity": "medium",
        "optional": False,
    },
    {
        "name": "PCA",
        "key": "pca",
        "family": ModelFamily.UNSUPERVISED,
        "tasks": ("dimensionality_reduction",),
        "explainable": True,
        "complexity": "low",
        "optional": False,
    },
)


class PCAProjector(BaseEstimator, TransformerMixin):
    """PCA with predict() = transform() so the shared sklearn Pipeline can score new rows."""

    def __init__(self, n_components: int = 2, random_state: int = 0) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        n = min(int(self.n_components), arr.shape[0], arr.shape[1])
        n = max(1, n)
        self.n_components_ = n
        self._pca = PCA(n_components=n, random_state=self.random_state)
        self._pca.fit(arr)
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        return self

    def transform(self, X):
        return self._pca.transform(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.transform(X)

    def inverse_transform(self, Z):
        return self._pca.inverse_transform(Z)


def build_unsupervised_estimator(
    estimator_key: str,
    problem_type: ProblemType,
    random_state: int,
    params: dict[str, Any] | None = None,
):
    params = dict(params or {})
    if estimator_key == "kmeans":
        return KMeans(
            n_clusters=int(params.pop("n_clusters", 3)),
            n_init=int(params.pop("n_init", 10)),
            random_state=random_state,
            **params,
        )
    if estimator_key == "gmm":
        n = int(params.pop("n_clusters", params.pop("n_components", 3)))
        return GaussianMixture(
            n_components=n,
            covariance_type=params.pop("covariance_type", "diag"),
            random_state=random_state,
            **params,
        )
    if estimator_key == "birch":
        return Birch(n_clusters=int(params.pop("n_clusters", 3)), **params)
    if estimator_key == "isolation_forest":
        contamination = params.pop("contamination", "auto")
        return IsolationForest(
            n_estimators=int(params.pop("n_estimators", 80)),
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,
            **params,
        )
    if estimator_key == "one_class_svm":
        return OneClassSVM(
            nu=float(params.pop("nu", 0.1)),
            gamma=params.pop("gamma", "scale"),
            kernel=params.pop("kernel", "rbf"),
            **params,
        )
    if estimator_key == "local_outlier_factor":
        return LocalOutlierFactor(
            n_neighbors=int(params.pop("n_neighbors", 20)),
            contamination=params.pop("contamination", "auto"),
            novelty=True,
            n_jobs=1,
            **params,
        )
    if estimator_key == "pca":
        return PCAProjector(
            n_components=int(params.pop("n_components", 2)),
            random_state=random_state,
        )
    raise KeyError(f"Estimator '{estimator_key}' is not available for {problem_type.value}")
