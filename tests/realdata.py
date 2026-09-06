"""Load real, published datasets as DataFrames. No synthetic generators here."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine


def breast_cancer_frame() -> pd.DataFrame:
    bundle = load_breast_cancer(as_frame=True)
    frame = bundle.frame.copy()
    frame = frame.rename(columns={"target": "diagnosis"})
    return frame


def iris_frame() -> pd.DataFrame:
    bundle = load_iris(as_frame=True)
    frame = bundle.frame.copy()
    frame = frame.rename(columns={"target": "species"})
    return frame


def wine_frame() -> pd.DataFrame:
    bundle = load_wine(as_frame=True)
    frame = bundle.frame.copy()
    frame = frame.rename(columns={"target": "cultivar"})
    return frame


def diabetes_frame() -> pd.DataFrame:
    bundle = load_diabetes(as_frame=True)
    frame = bundle.frame.copy()
    frame = frame.rename(columns={"target": "disease_progression"})
    return frame
