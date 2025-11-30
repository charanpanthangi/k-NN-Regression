"""Utilities to load the California Housing dataset for k-NN regression."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_california_housing() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the California Housing dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix ``X`` as a DataFrame and target vector ``y`` as a Series.
    """

    dataset = fetch_california_housing(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y
