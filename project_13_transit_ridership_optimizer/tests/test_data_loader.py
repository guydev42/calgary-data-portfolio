"""Tests for transit ridership optimizer data loader."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import preprocess_ridership, preprocess_stops, engineer_features


@pytest.fixture
def sample_ridership_df():
    """Create a minimal ridership dataframe."""
    return pd.DataFrame({
        "year": [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022,
                 2023, 2023, 2023],
        "month": list(range(1, 13)),
        "ridership": [100000, 95000, 110000, 120000, 115000, 130000,
                      125000, 140000, 135000, 145000, 150000, 155000],
    })


@pytest.fixture
def sample_stops_df():
    """Create a minimal transit stops dataframe."""
    return pd.DataFrame({
        "stop_id": [1, 2, 3, 4],
        "stop_lat": [51.04, 51.05, 51.06, 51.07],
        "stop_lon": [-114.06, -114.07, -114.08, -114.09],
        "route_name": ["Route1", "Route1", "Route2", "Route2"],
        "stop_name": ["Stop A", "Stop B", "Stop C", "Stop D"],
    })


def test_preprocess_ridership_creates_date_column(sample_ridership_df):
    """preprocess_ridership should create date and quarter columns."""
    result = preprocess_ridership(sample_ridership_df)
    assert isinstance(result, pd.DataFrame)
    assert "date" in result.columns
    assert "quarter" in result.columns
    assert "ridership" in result.columns


def test_preprocess_stops_standardizes_coords(sample_stops_df):
    """preprocess_stops should standardize latitude/longitude columns."""
    result = preprocess_stops(sample_stops_df)
    assert isinstance(result, pd.DataFrame)
    assert "latitude" in result.columns
    assert "longitude" in result.columns


def test_engineer_features_adds_lags(sample_ridership_df):
    """engineer_features should add lag and rolling mean columns."""
    preprocessed = preprocess_ridership(sample_ridership_df)
    result = engineer_features(preprocessed)
    assert isinstance(result, pd.DataFrame)
    assert "lag_1m" in result.columns
    assert "rolling_mean_3m" in result.columns
    assert "yoy_change" in result.columns
