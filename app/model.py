"""Model utilities for k-NN regression."""

from typing import Tuple

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def build_knn_regressor(
    n_neighbors: int = 5, weights: str = "distance", metric: str = "euclidean"
) -> Pipeline:
    """
    Create a pipeline with scaling and k-NN regression.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of nearest neighbors to average. Small ``k`` captures local patterns
        but can be noisy; larger ``k`` smooths predictions but may miss details.
    weights : str, optional
        ``"uniform"`` weighs all neighbors equally; ``"distance"`` weighs closer
        neighbors more heavily.
    metric : str, optional
        Distance metric for neighbor search. Euclidean is the common default.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline combining ``StandardScaler`` and ``KNeighborsRegressor``.
    """

    # Scaling ensures all features contribute fairly to the distance calculation.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                ),
            ),
        ]
    )
    return pipeline
