"""Tests for project_25 model module."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import _get_models_and_grids


def _make_synthetic_data():
    """Create minimal synthetic train/test arrays."""
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 200, 50, 18
    X_train = rng.randn(n_train, n_features)
    y_train = rng.choice([0, 1], n_train, p=[0.75, 0.25])
    X_test = rng.randn(n_test, n_features)
    y_test = rng.choice([0, 1], n_test, p=[0.75, 0.25])
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X_train, X_test, y_train, y_test, feature_names


def test_get_models_and_grids_returns_dict():
    """_get_models_and_grids should return at least LR and RF."""
    models = _get_models_and_grids()
    assert isinstance(models, dict)
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    for name, config in models.items():
        assert "model" in config
        assert "params" in config
        assert "needs_scaling" in config


def test_logistic_regression_trains():
    """Logistic Regression from the model config should fit and predict."""
    from sklearn.preprocessing import StandardScaler
    models = _get_models_and_grids()
    X_train, X_test, y_train, y_test, _ = _make_synthetic_data()
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    lr = models["Logistic Regression"]["model"]
    lr.fit(X_tr, y_train)
    preds = lr.predict(X_te)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})


def test_random_forest_predicts_proba():
    """Random Forest from the model config should produce valid probabilities."""
    models = _get_models_and_grids()
    X_train, X_test, y_train, y_test, _ = _make_synthetic_data()
    rf = models["Random Forest"]["model"]
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)
    assert proba.shape == (len(y_test), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
