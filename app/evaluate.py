"""Evaluation metrics for regression models."""

from math import sqrt
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute common regression metrics.

    Returns
    -------
    Dict[str, float]
        Dictionary with MSE, MAE, RMSE, and R2 scores.
    """

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
