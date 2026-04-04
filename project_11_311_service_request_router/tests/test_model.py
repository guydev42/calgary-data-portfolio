"""Tests for 311 service request router model."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import prepare_model_data, train_models, get_feature_importance


@pytest.fixture
def sample_engineered_df():
    """Create a dataframe that mimics output of engineer_features."""
    n = 200
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "channel": rng.choice(["Phone", "Web", "App"], n),
        "ward": rng.choice(["W01", "W02", "W03"], n),
        "hour": rng.randint(0, 24, n),
        "day_of_week": rng.randint(0, 7, n),
        "month": rng.randint(1, 13, n),
        "year": rng.choice([2022, 2023], n),
        "community_request_count": rng.randint(10, 500, n),
        "community_avg_resolution": rng.uniform(1, 100, n),
        "service_type_frequency": rng.randint(5, 200, n),
        "service_request_type": rng.choice(["Pothole", "Noise", "Graffiti"], n),
        "agency_responsible": rng.choice(["Roads", "Bylaw", "Parks"], n),
    })


def test_prepare_model_data_returns_correct_types(sample_engineered_df):
    """prepare_model_data should return X, y, encoders, feature_names."""
    X, y, encoders, features = prepare_model_data(sample_engineered_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(encoders, dict)
    assert isinstance(features, list)
    assert len(X) == len(y)


def test_train_models_returns_results(sample_engineered_df):
    """train_models should return trained models and metric dicts."""
    X, y, _, _ = prepare_model_data(sample_engineered_df)
    trained, results, scaler, X_test, y_test = train_models(X, y)
    assert isinstance(results, dict)
    assert len(results) > 0
    for name, metrics in results.items():
        assert "Accuracy" in metrics
        assert "Weighted F1" in metrics


def test_get_feature_importance_returns_dataframe(sample_engineered_df):
    """get_feature_importance should return a DataFrame for tree models."""
    X, y, _, features = prepare_model_data(sample_engineered_df)
    trained, _, _, _, _ = train_models(X, y)
    rf_model = trained["Random Forest"]
    importance = get_feature_importance(rf_model, features)
    assert isinstance(importance, pd.DataFrame)
    assert "Feature" in importance.columns
    assert "Importance" in importance.columns
