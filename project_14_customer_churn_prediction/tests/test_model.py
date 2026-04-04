"""Tests for customer churn prediction model."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import _get_models_and_grids


@pytest.fixture
def sample_train_data():
    """Create synthetic train/test data for churn modeling."""
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 200, 50, 10
    X_train = rng.randn(n_train, n_features)
    y_train = rng.choice([0, 1], n_train, p=[0.73, 0.27])
    X_test = rng.randn(n_test, n_features)
    y_test = rng.choice([0, 1], n_test, p=[0.73, 0.27])
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X_train, X_test, y_train, y_test, feature_names


def test_get_models_and_grids_returns_dict():
    """_get_models_and_grids should return a dict of model configs."""
    models = _get_models_and_grids()
    assert isinstance(models, dict)
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    for name, config in models.items():
        assert "model" in config
        assert "params" in config
        assert "needs_scaling" in config


def test_logistic_regression_fits_and_predicts(sample_train_data):
    """Logistic Regression should fit and predict without error."""
    X_train, X_test, y_train, y_test, _ = sample_train_data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = _get_models_and_grids()
    lr = models["Logistic Regression"]["model"]
    lr.fit(X_train_s, y_train)
    preds = lr.predict(X_test_s)
    assert preds.shape == (X_test.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_random_forest_fits_and_predicts(sample_train_data):
    """Random Forest should fit and predict without error."""
    X_train, X_test, y_train, y_test, _ = sample_train_data
    models = _get_models_and_grids()
    rf = models["Random Forest"]["model"]
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert preds.shape == (X_test.shape[0],)
    assert hasattr(rf, "feature_importances_")
