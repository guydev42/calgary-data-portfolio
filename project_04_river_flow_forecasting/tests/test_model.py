"""Tests for project_04 model module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess, resample_daily
from model import temporal_train_test_split, train_random_forest, evaluate


@pytest.fixture
def daily_df():
    """Create a synthetic daily river flow DataFrame with features."""
    np.random.seed(42)
    n = 5000  # enough 5-min intervals for multiple days
    timestamps = pd.date_range("2023-01-01", periods=n, freq="5min")
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "level": np.random.uniform(1.0, 3.0, n),
        "flow_rate": np.random.uniform(50, 200, n),
    })
    preprocessed = preprocess(raw)
    return resample_daily(preprocessed)


def test_temporal_split(daily_df):
    """temporal_train_test_split should split data in temporal order."""
    target = "flow_rate" if "flow_rate" in daily_df.columns else "level"
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        daily_df, target_col=target
    )
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) > len(X_test)


def test_train_random_forest(daily_df):
    """train_random_forest should return a fitted model that can predict."""
    target = "flow_rate" if "flow_rate" in daily_df.columns else "level"
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        daily_df, target_col=target
    )
    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_evaluate_returns_valid_metrics(daily_df):
    """evaluate should return a dict with finite MAE, RMSE, R2."""
    target = "flow_rate" if "flow_rate" in daily_df.columns else "level"
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        daily_df, target_col=target
    )
    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test.values, preds)
    assert np.isfinite(metrics["MAE"])
    assert np.isfinite(metrics["RMSE"])
    assert np.isfinite(metrics["R2"])
