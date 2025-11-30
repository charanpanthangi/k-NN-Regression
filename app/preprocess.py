"""Preprocessing utilities: train/test split and scaling.

Because k-NN is distance-based, feature scaling is mandatory; otherwise
features with larger numeric ranges dominate the distance calculation.
"""

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessResult:
    """Container for preprocessed datasets."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler


def split_and_scale(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> PreprocessResult:
    """
    Split the data and fit a StandardScaler.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, optional
        Proportion of the dataset to include in the test split, by default 0.2.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    PreprocessResult
        Dataclass containing train/test splits and the fitted scaler.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    return PreprocessResult(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
    )
