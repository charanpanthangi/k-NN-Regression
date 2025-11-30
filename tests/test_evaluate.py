import numpy as np

from app.evaluate import regression_metrics


def test_regression_metrics_output():
    y_true = np.array([3.0, 4.5, 5.0])
    y_pred = np.array([2.5, 4.0, 5.5])

    metrics = regression_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"mse", "mae", "rmse", "r2"}
    assert metrics["mse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["r2"] <= 1
