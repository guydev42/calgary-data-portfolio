"""Tests for project_08 data_loader module."""

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


@pytest.fixture
def sample_production_df():
    """Create a synthetic solar production DataFrame."""
    return _generate_synthetic_production()


def test_preprocess_returns_dataframe(sample_production_df):
    """preprocess_production should return a DataFrame with temporal features."""
    result = preprocess_production(sample_production_df)
    assert isinstance(result, pd.DataFrame)
    assert "year" in result.columns
    assert "month" in result.columns
    assert "month_sin" in result.columns


def test_preprocess_numeric_production(sample_production_df):
    """preprocess_production should ensure solar_pv_production_kwh is numeric."""
    result = preprocess_production(sample_production_df)
    assert pd.api.types.is_numeric_dtype(result["solar_pv_production_kwh"])


def test_rolling_and_lag_features(sample_production_df):
    """add_rolling_features and add_lag_features should create expected columns."""
    processed = preprocess_production(sample_production_df)
    processed = add_rolling_features(processed)
    processed = add_lag_features(processed)
    assert "rolling_avg_3m" in processed.columns
    assert "rolling_avg_12m" in processed.columns
    assert "lag_1m" in processed.columns
    assert "lag_12m" in processed.columns
