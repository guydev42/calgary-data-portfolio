"""Tests for project_08 model module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import (
    preprocess_production,
    add_rolling_features,
    add_lag_features,
    _generate_synthetic_production,
)
from model import (
    prepare_features,
    temporal_train_test_split,
    compute_metrics,
    get_models,
)


@pytest.fixture
def feature_df():
    """Create a fully featured solar production DataFrame."""
    raw = _generate_synthetic_production()
    df = preprocess_production(raw)
    df = add_rolling_features(df)
    df = add_lag_features(df)
    return df


def test_prepare_features(feature_df):
    """prepare_features should return X, y, and cleaned df with matching lengths."""
    X, y, clean = prepare_features(feature_df)
    assert len(X) == len(y)
    assert len(X) > 0
    assert isinstance(X, pd.DataFrame)


def test_temporal_train_test_split(feature_df):
    """temporal_train_test_split should split by date, with train before test."""
    train, test = temporal_train_test_split(feature_df, test_months=6)
    assert len(train) > 0
    assert len(test) > 0
    assert train["period_dt"].max() <= test["period_dt"].min()


def test_model_training_and_metrics(feature_df):
    """Models should train and compute_metrics should return finite values."""
    train, test = temporal_train_test_split(feature_df, test_months=6)
    X_train, y_train, _ = prepare_features(train)
    X_test, y_test, _ = prepare_features(test)

    models = get_models()
    model_name = list(models.keys())[0]  # use first available model
    model = models[model_name]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = compute_metrics(y_test.values, preds)
    assert np.isfinite(metrics["MAE"])
    assert np.isfinite(metrics["RMSE"])
    assert np.isfinite(metrics["R2"])
