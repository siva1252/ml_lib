"""Thin adapters over sklearn and optional boosting libraries. No training logic here."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlverdict.core.enums import ProblemType


def _optional_constructor(module: str, class_name: str):
    try:
        mod = __import__(module, fromlist=[class_name])
        return getattr(mod, class_name)
    except Exception:
        return None


def available_boosters() -> dict[str, bool]:
    return {
        "xgboost": _optional_constructor("xgboost", "XGBClassifier") is not None,
        "lightgbm": _optional_constructor("lightgbm", "LGBMClassifier") is not None,
        "catboost": _optional_constructor("catboost", "CatBoostClassifier") is not None,
    }


def build_estimator(
    estimator_key: str,
    problem_type: ProblemType,
    random_state: int,
    params: dict[str, Any] | None = None,
):
    from mlverdict.models.unsupervised import UNSUPERVISED_TYPES, build_unsupervised_estimator

    if problem_type in UNSUPERVISED_TYPES:
        return build_unsupervised_estimator(estimator_key, problem_type, random_state, params)

    params = dict(params or {})
    cls = problem_type != ProblemType.REGRESSION

    registry: dict[str, Any] = {}
    if cls:
        registry["logistic_regression"] = lambda: LogisticRegression(
            max_iter=400, solver="lbfgs", random_state=random_state, **params
        )
        registry["decision_tree"] = lambda: DecisionTreeClassifier(
            max_depth=params.pop("max_depth", 8),
            min_samples_leaf=params.pop("min_samples_leaf", 2),
            random_state=random_state,
            **params,
        )
        registry["knn"] = lambda: KNeighborsClassifier(
            n_neighbors=params.pop("n_neighbors", 15),
            weights=params.pop("weights", "distance"),
            **params,
        )
        registry["svm"] = lambda: SVC(
            C=params.pop("C", 1.0),
            gamma=params.pop("gamma", "scale"),
            probability=True,
            random_state=random_state,
            **params,
        )
        registry["gaussian_nb"] = lambda: GaussianNB(
            var_smoothing=params.pop("var_smoothing", 1e-9),
            **params,
        )
        registry["random_forest"] = lambda: RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 80),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=params.pop("min_samples_leaf", 1),
            n_jobs=1,
            random_state=random_state,
            **params,
        )
        registry["extra_trees"] = lambda: ExtraTreesClassifier(
            n_estimators=params.pop("n_estimators", 80),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=params.pop("min_samples_leaf", 1),
            n_jobs=1,
            random_state=random_state,
            **params,
        )
        registry["hist_gradient_boosting"] = lambda: HistGradientBoostingClassifier(
            max_depth=params.pop("max_depth", 6),
            learning_rate=params.pop("learning_rate", 0.1),
            max_iter=params.pop("max_iter", 80),
            random_state=random_state,
            **params,
        )
        xgb_c = _optional_constructor("xgboost", "XGBClassifier")
        lgb_c = _optional_constructor("lightgbm", "LGBMClassifier")
        cat_c = _optional_constructor("catboost", "CatBoostClassifier")
        if xgb_c:
            registry["xgboost"] = lambda: xgb_c(
                n_estimators=params.pop("n_estimators", 80),
                max_depth=params.pop("max_depth", 4),
                learning_rate=params.pop("learning_rate", 0.1),
                n_jobs=1,
                verbosity=0,
                random_state=random_state,
                **params,
            )
        if lgb_c:
            registry["lightgbm"] = lambda: lgb_c(
                n_estimators=params.pop("n_estimators", 80),
                max_depth=params.pop("max_depth", -1),
                learning_rate=params.pop("learning_rate", 0.1),
                verbose=-1,
                random_state=random_state,
                **params,
            )
        if cat_c:
            registry["catboost"] = lambda: cat_c(
                iterations=params.pop("n_estimators", params.pop("iterations", 80)),
                depth=params.pop("max_depth", params.pop("depth", 4)),
                learning_rate=params.pop("learning_rate", 0.1),
                verbose=False,
                random_seed=random_state,
                allow_writing_files=False,
                **params,
            )
    else:
        registry["ridge"] = lambda: Ridge(random_state=random_state, **params)
        registry["linear_regression"] = lambda: LinearRegression(**params)
        registry["lasso"] = lambda: Lasso(
            alpha=params.pop("alpha", 0.01), max_iter=5000, random_state=random_state, **params
        )
        registry["elastic_net"] = lambda: ElasticNet(
            alpha=params.pop("alpha", 0.01),
            l1_ratio=params.pop("l1_ratio", 0.5),
            max_iter=5000,
            random_state=random_state,
            **params,
        )
        registry["decision_tree"] = lambda: DecisionTreeRegressor(
            max_depth=params.pop("max_depth", 8),
            min_samples_leaf=params.pop("min_samples_leaf", 2),
            random_state=random_state,
            **params,
        )
        registry["knn"] = lambda: KNeighborsRegressor(
            n_neighbors=params.pop("n_neighbors", 15),
            weights=params.pop("weights", "distance"),
            **params,
        )
        registry["svm"] = lambda: SVR(
            C=params.pop("C", 1.0),
            gamma=params.pop("gamma", "scale"),
            epsilon=params.pop("epsilon", 0.1),
            **params,
        )
        registry["random_forest"] = lambda: RandomForestRegressor(
            n_estimators=params.pop("n_estimators", 80),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=params.pop("min_samples_leaf", 1),
            n_jobs=1,
            random_state=random_state,
            **params,
        )
        registry["extra_trees"] = lambda: ExtraTreesRegressor(
            n_estimators=params.pop("n_estimators", 80),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=params.pop("min_samples_leaf", 1),
            n_jobs=1,
            random_state=random_state,
            **params,
        )
        registry["hist_gradient_boosting"] = lambda: HistGradientBoostingRegressor(
            max_depth=params.pop("max_depth", 6),
            learning_rate=params.pop("learning_rate", 0.1),
            max_iter=params.pop("max_iter", 80),
            random_state=random_state,
            **params,
        )
        xgb_r = _optional_constructor("xgboost", "XGBRegressor")
        lgb_r = _optional_constructor("lightgbm", "LGBMRegressor")
        cat_r = _optional_constructor("catboost", "CatBoostRegressor")
        if xgb_r:
            registry["xgboost"] = lambda: xgb_r(
                n_estimators=params.pop("n_estimators", 80),
                max_depth=params.pop("max_depth", 4),
                learning_rate=params.pop("learning_rate", 0.1),
                n_jobs=1,
                verbosity=0,
                random_state=random_state,
                **params,
            )
        if lgb_r:
            registry["lightgbm"] = lambda: lgb_r(
                n_estimators=params.pop("n_estimators", 80),
                learning_rate=params.pop("learning_rate", 0.1),
                verbose=-1,
                random_state=random_state,
                **params,
            )
        if cat_r:
            registry["catboost"] = lambda: cat_r(
                iterations=params.pop("n_estimators", 80),
                depth=params.pop("max_depth", 4),
                learning_rate=params.pop("learning_rate", 0.1),
                verbose=False,
                random_seed=random_state,
                allow_writing_files=False,
                **params,
            )

    if estimator_key not in registry:
        raise KeyError(f"Estimator '{estimator_key}' is not available for {problem_type.value}")
    return registry[estimator_key]()
