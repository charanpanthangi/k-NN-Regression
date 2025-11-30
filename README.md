# k-Nearest Neighbors (k-NN) Regression Tutorial

A beginner-friendly walkthrough of k-NN Regression using the California Housing dataset. The project shows how to load data, scale features, train a k-NN regressor, evaluate it with common metrics, and visualize results.

## What is k-NN Regression?
- **Idea:** Predict a value by averaging the targets of the closest training points in feature space.
- **Distance-based:** Uses a distance metric (default Euclidean) to find neighbors.
- **Key hyperparameters:**
  - `n_neighbors`: how many neighbors to average. Small `k` captures local patterns but can be noisy; larger `k` smooths predictions.
  - `weights`: `"uniform"` (all neighbors equal) or `"distance"` (closer neighbors matter more).
  - `metric`: distance function, typically Euclidean.
- **Why scaling matters:** Features with larger numeric ranges dominate distance calculations. Standardizing keeps each feature on comparable scale.

## When to use / when not to use
- ✅ Works well when the relationship is locally smooth and the dataset is not extremely large.
- ✅ Easy to understand and quick to prototype.
- ❌ Can be slow on very large datasets because every prediction scans the training set.
- ❌ Sensitive to irrelevant or unscaled features.

## Dataset
The [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) from scikit-learn. It includes housing features (median income, latitude, etc.) and the median house value target.

## Project structure
```
app/
  data.py        # Load the California Housing dataset
  preprocess.py  # Train/test split and scaling with StandardScaler
  model.py       # k-NN regressor inside a Pipeline
  evaluate.py    # MSE, MAE, RMSE, R² metrics
  visualize.py   # Predicted vs Actual and Residual plots (SVG)
  main.py        # End-to-end pipeline
notebooks/
  demo_knn_regression.ipynb  # Notebook demo with explanations and plots
tests/
  test_data.py, test_model.py, test_evaluate.py
examples/
  README_examples.md
requirements.txt
Dockerfile
LICENSE (MIT)
```

## Step-by-step pipeline
1. **Load data:** `fetch_california_housing` returns features and target.
2. **Split & scale:** Train/test split plus `StandardScaler` so all features contribute fairly to distance.
3. **Build model:** `KNeighborsRegressor` with distance weighting inside a `Pipeline`.
4. **Train:** Fit the pipeline on training data.
5. **Evaluate:** MSE, MAE, RMSE, R² on the test set.
6. **Visualize:** Predicted vs actual scatter and residual distribution saved as SVG.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install -r requirements.txt
python app/main.py
```

To run the notebook:
```bash
jupyter notebook notebooks/demo_knn_regression.ipynb
```

## Future improvements
- Hyperparameter search for the best `k` with `GridSearchCV`.
- Try alternative distance metrics (e.g., Manhattan, Minkowski).
- Experiment with weighting strategies and feature selection to reduce noise.

## License
MIT License. See [LICENSE](LICENSE) for details.
