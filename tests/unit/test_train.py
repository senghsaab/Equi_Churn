"""
tests/test_train.py — Unit Tests for Training Module (Stage 6)
=================================================================
"""

import numpy as np
import pandas as pd
import pytest

from src.train import (
    MODEL_REGISTRY,
    scale_features,
    split_data,
    train_all_models,
    train_single_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_X_y():
    """Create a small synthetic feature matrix and target."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        {
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.rand(n),
        }
    )
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.7, 0.3]), name="churn_flag")
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSplitData:
    def test_split_sizes(self, sample_X_y):
        X, y = sample_X_y
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) + len(X_test) == len(X)
        assert abs(y_train.mean() - y_test.mean()) < 0.1  # stratified

    def test_stratification(self, sample_X_y):
        X, y = sample_X_y
        _, _, y_train, y_test = split_data(X, y)
        assert abs(y_train.mean() - y.mean()) < 0.05


class TestScaleFeatures:
    def test_scaled_mean_near_zero(self, sample_X_y):
        X, y = sample_X_y
        X_train, X_test, _, _ = split_data(X, y)
        X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
        assert abs(X_train_s.mean().mean()) < 0.1


class TestTraining:
    def test_single_model_dummy(self, sample_X_y):
        """DummyClassifier trains without search."""
        X, y = sample_X_y
        model, best_params = train_single_model("dummy", X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert best_params is None  # no hyperparameter search

    def test_single_model_logistic(self, sample_X_y):
        """LogisticRegression trains without search."""
        X, y = sample_X_y
        model, best_params = train_single_model("logistic_regression", X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})
        assert best_params is None

    def test_single_model_random_forest_with_gridsearch(self, sample_X_y):
        """RandomForest uses GridSearchCV and returns best_params."""
        X, y = sample_X_y
        model, best_params = train_single_model("random_forest", X, y)
        assert best_params is not None
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "min_samples_split" in best_params
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_single_model_gradient_boosting_with_randomsearch(self, sample_X_y):
        """GradientBoosting uses RandomizedSearchCV and returns best_params."""
        X, y = sample_X_y
        model, best_params = train_single_model("gradient_boosting", X, y)
        assert best_params is not None
        assert "n_estimators" in best_params
        assert "learning_rate" in best_params
        assert "max_depth" in best_params

    def test_all_models_train(self, sample_X_y):
        X, y = sample_X_y
        models, all_params = train_all_models(X, y)
        assert len(models) == len(MODEL_REGISTRY)
        # RF and GB should have best_params
        assert all_params["random_forest"] is not None
        assert all_params["gradient_boosting"] is not None
        # Dummy and LR should not
        assert all_params["dummy"] is None
        assert all_params["logistic_regression"] is None

    def test_unknown_model_raises(self, sample_X_y):
        X, y = sample_X_y
        with pytest.raises(ValueError):
            train_single_model("nonexistent_model", X, y)
