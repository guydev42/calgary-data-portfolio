"""Tests for project_04 data_loader module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess, resample_daily


@pytest.fixture
def sample_river_df():
    """Create a synthetic river flow DataFrame."""
    np.random.seed(42)
    n = 1000
    timestamps = pd.date_range("2023-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": timestamps,
        "level": np.random.uniform(1.0, 3.0, n),
        "flow_rate": np.random.uniform(50, 200, n),
    })


def test_preprocess_returns_dataframe(sample_river_df):
    """preprocess should return a DataFrame with temporal features."""
    result = preprocess(sample_river_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_preprocess_temporal_columns(sample_river_df):
    """preprocess should add hour, day_of_week, month, year columns."""
    result = preprocess(sample_river_df)
    for col in ["hour", "day_of_week", "month", "year"]:
        assert col in result.columns


def test_resample_daily_creates_lag_features(sample_river_df):
    """resample_daily should produce daily data with rolling and lag features."""
    preprocessed = preprocess(sample_river_df)
    daily = resample_daily(preprocessed)
    assert isinstance(daily, pd.DataFrame)
    assert "date" in daily.columns
    lag_cols = [c for c in daily.columns if "lag_" in c]
    assert len(lag_cols) > 0
