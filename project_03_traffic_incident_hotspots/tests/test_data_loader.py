"""Tests for project_03 data_loader module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess_dataframe, create_clustering_features


@pytest.fixture
def sample_traffic_df():
    """Create a synthetic traffic incidents DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "start_dt": pd.date_range("2023-01-01", periods=n, freq="h"),
        "quadrant": np.random.choice(["NW", "NE", "SW", "SE"], n),
        "latitude": np.random.uniform(50.9, 51.2, n),
        "longitude": np.random.uniform(-114.3, -113.9, n),
        "count": np.random.randint(1, 5, n),
    })


def test_preprocess_returns_dataframe(sample_traffic_df):
    """preprocess_dataframe should return a DataFrame with temporal features."""
    result = preprocess_dataframe(sample_traffic_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_preprocess_creates_temporal_columns(sample_traffic_df):
    """preprocess_dataframe should extract hour, day_of_week, month, year."""
    result = preprocess_dataframe(sample_traffic_df)
    for col in ["hour", "day_of_week", "month", "year"]:
        assert col in result.columns, f"Missing column: {col}"


def test_create_clustering_features_shape(sample_traffic_df):
    """create_clustering_features should return an (n, 2) array of lat/lon."""
    processed = preprocess_dataframe(sample_traffic_df)
    features = create_clustering_features(processed)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert features.shape[1] == 2
