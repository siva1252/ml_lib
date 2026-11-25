"""Unsupervised estimators. predict() must work on new rows (no agglomerative)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin, TransformerMixin
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
        "name": "Detector Ensemble",
        "key": "detector_ensemble",
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
    """PCA with predict() = transform() so the shared sklearn Pipeline can score new rows.

    ``n_components`` in (0, 1) keeps the fewest components that explain that share of variance;
    an integer keeps exactly that many (capped by the data shape).
    """

    def __init__(self, n_components: float = 0.95, random_state: int = 0) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        limit = min(arr.shape[0], arr.shape[1])
        if isinstance(self.n_components, float) and 0.0 < self.n_components < 1.0:
            target: float | int = float(self.n_components)
            self._pca = PCA(n_components=target, svd_solver="full", random_state=self.random_state)
        else:
            target = max(1, min(int(self.n_components), limit))
            self._pca = PCA(n_components=target, random_state=self.random_state)
        self._pca.fit(arr)
        self.n_components_ = int(self._pca.n_components_)
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        return self

    def transform(self, X):
        return self._pca.transform(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.transform(X)

    def inverse_transform(self, Z):
        return self._pca.inverse_transform(np.asarray(Z, dtype=float))


class DetectorEnsemble(OutlierMixin, BaseEstimator):
    """Max-vote ensemble of Isolation Forest, One-Class SVM and Local Outlier Factor.

    Each detector's score is turned into a percentile against its own training scores; a row's
    ensemble normality is the *lowest* percentile any detector gives it, so a row is flagged when
    any detector finds it extreme. In MLVerdict's benchmarks this stayed close to the best single
    detector on every outlier shape (global, clustered, structural), where each single detector
    fails badly on at least one. One-Class SVM is skipped above ``max_rows_for_svm`` rows.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_neighbors: int = 20,
        max_rows_for_svm: int = 10_000,
        random_state: int = 0,
    ) -> None:
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.max_rows_for_svm = max_rows_for_svm
        self.random_state = random_state

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        n = arr.shape[0]
        forest = IsolationForest(n_estimators=100, random_state=self.random_state, n_jobs=1).fit(arr)
        lof = LocalOutlierFactor(n_neighbors=max(1, min(self.n_neighbors, n - 1)), novelty=True).fit(arr)
        self.detectors_ = [("isolation_forest", forest), ("local_outlier_factor", lof)]
        # LOF's leave-self-out training scores are the unbiased reference for its percentiles.
        self.references_ = [np.sort(forest.score_samples(arr)), np.sort(lof.negative_outlier_factor_)]
        if n <= self.max_rows_for_svm:
            svm = OneClassSVM(nu=self.contamination, gamma="scale").fit(arr)
            self.detectors_.append(("one_class_svm", svm))
            self.references_.append(np.sort(svm.score_samples(arr)))
        self.threshold_ = float(np.quantile(self._combined(arr, training=True, lof_train=lof), self.contamination))
        return self

    def _combined(self, arr, *, training: bool = False, lof_train=None) -> np.ndarray:
        ranks = []
        for (name, det), ref in zip(self.detectors_, self.references_, strict=True):
            if training and name == "local_outlier_factor" and lof_train is not None:
                scores = lof_train.negative_outlier_factor_
            else:
                scores = det.score_samples(arr)
            ranks.append(np.searchsorted(ref, scores, side="right") / len(ref))
        return np.min(np.vstack(ranks), axis=0)

    def score_samples(self, X):
        return self._combined(np.asarray(X, dtype=float))

    def decision_function(self, X):
        return self.score_samples(X) - self.threshold_

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)


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
        contamination = params.pop("contamination", 0.05)
        return IsolationForest(
            n_estimators=int(params.pop("n_estimators", 80)),
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,
            **params,
        )
    if estimator_key == "one_class_svm":
        return OneClassSVM(
            nu=float(params.pop("nu", 0.05)),
            gamma=params.pop("gamma", "scale"),
            kernel=params.pop("kernel", "rbf"),
            **params,
        )
    if estimator_key == "local_outlier_factor":
        return LocalOutlierFactor(
            n_neighbors=int(params.pop("n_neighbors", 20)),
            contamination=params.pop("contamination", 0.05),
            novelty=True,
            n_jobs=1,
            **params,
        )
    if estimator_key == "detector_ensemble":
        return DetectorEnsemble(
            contamination=float(params.pop("contamination", 0.05)),
            max_rows_for_svm=int(params.pop("max_rows_for_svm", 10_000)),
            random_state=random_state,
            **params,
        )
    if estimator_key == "pca":
        return PCAProjector(
            n_components=params.pop("n_components", 0.95),
            random_state=random_state,
        )
    raise KeyError(f"Estimator '{estimator_key}' is not available for {problem_type.value}")
