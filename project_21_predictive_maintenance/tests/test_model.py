"""Tests for project_21 model module."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_sensor_data
from src.model import _get_models, train_and_evaluate


def _make_splits():
    """Helper to create train/test data from synthetic sensor data."""
    from sklearn.model_selection import train_test_split
    df = generate_sensor_data(n_readings=500, n_machines=10, random_state=0)
    feature_cols = [c for c in df.columns if c not in ("failure_within_7days", "machine_id")]
    X = df[feature_cols].values.astype(float)
    y = df["failure_within_7days"].values
    feature_names = list(feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_names


def test_get_models_returns_dict():
    """_get_models should return a dict of model configs."""
    models = _get_models()
    assert isinstance(models, dict)
    assert len(models) >= 2
    for name, config in models.items():
        assert "model" in config
        assert "needs_scaling" in config


def test_train_and_evaluate_runs():
    """train_and_evaluate should run on synthetic data without error."""
    X_train, X_test, y_train, y_test, feature_names = _make_splits()
    results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)
    assert isinstance(results, dict)
    assert len(results) >= 2
    for name, r in results.items():
        assert "auc_roc" in r
        assert 0 <= r["auc_roc"] <= 1


def test_model_predictions_shape():
    """Each trained model should produce predictions of the right length."""
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test, _ = _make_splits()
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})
