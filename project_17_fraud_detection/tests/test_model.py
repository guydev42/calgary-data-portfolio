"""Tests for fraud detection model."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import _get_models


@pytest.fixture
def sample_fraud_data():
    """Create synthetic fraud detection train/test data."""
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 300, 75, 10
    X_train = rng.randn(n_train, n_features)
    y_train = rng.choice([0, 1], n_train, p=[0.98, 0.02])
    X_test = rng.randn(n_test, n_features)
    y_test = rng.choice([0, 1], n_test, p=[0.98, 0.02])
    return X_train, X_test, y_train, y_test


def test_get_models_returns_dict():
    """_get_models should return a dict of model configs."""
    models = _get_models()
    assert isinstance(models, dict)
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    for name, config in models.items():
        assert "model" in config
        assert "needs_scaling" in config


def test_logistic_regression_fits(sample_fraud_data):
    """Logistic Regression from _get_models should fit without error."""
    X_train, X_test, y_train, y_test = sample_fraud_data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = _get_models()
    lr = models["Logistic Regression"]["model"]
    lr.fit(X_train_s, y_train)
    preds = lr.predict(X_test_s)
    probs = lr.predict_proba(X_test_s)
    assert preds.shape == (X_test.shape[0],)
    assert probs.shape == (X_test.shape[0], 2)


def test_random_forest_fits(sample_fraud_data):
    """Random Forest from _get_models should fit without error."""
    X_train, X_test, y_train, y_test = sample_fraud_data
    models = _get_models()
    rf = models["Random Forest"]["model"]
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert preds.shape == (X_test.shape[0],)
    assert hasattr(rf, "feature_importances_")
