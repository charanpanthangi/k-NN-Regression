import numpy as np

from app.data import load_california_housing
from app.preprocess import split_and_scale
from app.model import build_knn_regressor


def test_model_trains_and_predicts():
    X, y = load_california_housing()
    data = split_and_scale(X, y, test_size=0.1, random_state=0)

    model = build_knn_regressor(n_neighbors=3)
    model.fit(data.X_train, data.y_train)
    preds = model.predict(data.X_test)

    assert preds.shape[0] == data.X_test.shape[0]
    assert np.isfinite(preds).all()
