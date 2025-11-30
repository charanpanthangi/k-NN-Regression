"""End-to-end pipeline for k-NN regression on the California Housing dataset."""

from __future__ import annotations

from app.data import load_california_housing
from app.preprocess import split_and_scale
from app.model import build_knn_regressor
from app.evaluate import regression_metrics
from app import visualize



def run_pipeline() -> None:
    """Run data loading, preprocessing, training, evaluation, and visualization."""

    X, y = load_california_housing()
    data = split_and_scale(X, y)

    model = build_knn_regressor(n_neighbors=5, weights="distance", metric="euclidean")
    model.fit(data.X_train, data.y_train)

    predictions = model.predict(data.X_test)
    metrics = regression_metrics(data.y_test, predictions)

    pred_plot = visualize.plot_predictions(data.y_test.values, predictions)
    residual_plot = visualize.plot_residuals(data.y_test.values, predictions)

    print("Evaluation metrics for k-NN Regression:")
    for name, value in metrics.items():
        print(f"- {name.upper()}: {value:.4f}")

    print("\nSaved plots:")
    print(f"- Predicted vs Actual: {pred_plot}")
    print(f"- Residuals: {residual_plot}")


if __name__ == "__main__":
    run_pipeline()
