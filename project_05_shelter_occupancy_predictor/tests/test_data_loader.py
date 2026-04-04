"""Tests for project_05 data_loader module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess, add_rolling_features


@pytest.fixture
def sample_shelter_df():
    """Create a synthetic shelter occupancy DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    shelters = ["Alpha House", "Calgary Drop-In"]
    rows = []
    for shelter in shelters:
        for d in dates:
            cap = 100
            overnight = np.random.randint(50, 110)
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "shelter": shelter,
                "sheltertype": "Emergency",
                "organization": "Org A",
                "capacity": cap,
                "overnight": overnight,
            })
    return pd.DataFrame(rows)


def test_preprocess_returns_dataframe(sample_shelter_df):
    """preprocess should return a DataFrame with occupancy_rate."""
    result = preprocess(sample_shelter_df)
    assert isinstance(result, pd.DataFrame)
    assert "occupancy_rate" in result.columns


def test_preprocess_temporal_features(sample_shelter_df):
    """preprocess should extract day_of_week, month, year columns."""
    result = preprocess(sample_shelter_df)
    for col in ["day_of_week", "month", "year"]:
        assert col in result.columns


def test_add_rolling_features(sample_shelter_df):
    """add_rolling_features should create rolling and lag columns."""
    preprocessed = preprocess(sample_shelter_df)
    result = add_rolling_features(preprocessed)
    assert "rolling_7d_occupancy" in result.columns
    assert "lag_1d_occupancy" in result.columns
