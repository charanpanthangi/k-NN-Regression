import pandas as pd

from app.data import load_california_housing


def test_load_california_housing_shapes():
    X, y = load_california_housing()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]
    # Known shape of the dataset
    assert X.shape[1] == 8
    assert y.ndim == 1
