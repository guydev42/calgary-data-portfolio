"""Tests for project_01 model module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess_data, engineer_features
from model import prepare_model_data, train_models


@pytest.fixture
def model_ready_df():
    """Create a DataFrame large enough for model training."""
    np.random.seed(42)
    n = 200
    communities = ["Beltline", "Downtown", "Kensington", "Bridgeland"]
    df = pd.DataFrame({
        "permittype": np.random.choice(["Building", "Electrical"], n),
        "permitclass": np.random.choice(["Residential", "Commercial"], n),
        "permitclassgroup": np.random.choice(["House", "Commercial"], n),
        "workclass": np.random.choice(["New", "Alteration"], n),
        "workclassgroup": np.random.choice(["New Construction", "Alteration"], n),
        "applieddate": pd.date_range("2020-01-01", periods=n, freq="D"),
        "housingunits": np.random.randint(0, 5, n),
        "estprojectcost": np.random.uniform(10000, 500000, n),
        "totalsqft": np.random.uniform(100, 5000, n),
        "communityname": np.random.choice(communities, n),
        "latitude": np.random.uniform(51.0, 51.1, n),
        "longitude": np.random.uniform(-114.1, -114.0, n),
    })
    df = preprocess_data(df)
    df = engineer_features(df)
    return df


def test_prepare_model_data_returns_correct_types(model_ready_df):
    """prepare_model_data should return X (DataFrame), y (Series), encoders, and feature names."""
    X, y, label_encoders, features = prepare_model_data(model_ready_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(X) > 0


def test_train_models_returns_results(model_ready_df):
    """train_models should return trained models and valid metric dictionaries."""
    X, y, _, _ = prepare_model_data(model_ready_df)
    trained_models, results, scaler, X_test, y_test = train_models(X, y)
    assert len(trained_models) > 0
    assert len(results) > 0
    for name, metrics in results.items():
        assert isinstance(metrics["R2"], float)
        assert isinstance(metrics["MAE"], float)
        assert np.isfinite(metrics["R2"])
