"""Visualization helpers for k-NN regression results."""

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, filename: str = "pred_vs_actual.svg") -> Path:
    """
    Create a predicted vs actual scatter plot and save as SVG.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values from the model.
    filename : str, optional
        Name of the file to save, by default "pred_vs_actual.svg".

    Returns
    -------
    Path
        Path to the saved plot file.
    """

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual")
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, filename: str = "residuals.svg") -> Path:
    """
    Plot residual distribution and save as SVG.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.
    filename : str, optional
        Name of the file to save, by default "residuals.svg".

    Returns
    -------
    Path
        Path to the saved plot file.
    """

    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True)
    plt.axvline(0, color="red", linestyle="--", label="Zero Error")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residual Distribution")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, format="svg")
    plt.close()
    return output_path
