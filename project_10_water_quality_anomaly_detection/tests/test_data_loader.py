"""Tests for water quality anomaly detection data loader."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import preprocess, pivot_parameters, add_zscore_features


@pytest.fixture
def sample_raw_df():
    """Create a minimal raw water quality dataframe."""
    return pd.DataFrame({
        "sample_site": ["Site_A"] * 6 + ["Site_B"] * 6,
        "sample_date": pd.date_range("2023-01-01", periods=6).tolist() * 2,
        "parameter": ["pH", "Temperature", "Turbidity"] * 4,
        "numeric_result": [7.2, 10.0, 3.5, 7.5, 11.0, 4.0,
                           6.8, 9.0, 2.5, 7.0, 10.5, 3.0],
        "latitude_degrees": [51.05] * 12,
        "longitude_degrees": [-114.07] * 12,
    })


def test_preprocess_returns_dataframe(sample_raw_df):
    """preprocess should return a DataFrame with expected columns."""
    result = preprocess(sample_raw_df)
    assert isinstance(result, pd.DataFrame)
    assert "year" in result.columns
    assert "month" in result.columns
    assert pd.api.types.is_datetime64_any_dtype(result["sample_date"])


def test_pivot_parameters_widens_data(sample_raw_df):
    """pivot_parameters should create one column per parameter."""
    cleaned = preprocess(sample_raw_df)
    pivoted = pivot_parameters(cleaned)
    assert isinstance(pivoted, pd.DataFrame)
    assert "sample_site" in pivoted.columns
    assert "sample_date" in pivoted.columns
    # At least one parameter should become a column
    assert any(col in pivoted.columns for col in ["pH", "Temperature", "Turbidity"])


def test_add_zscore_features_creates_columns(sample_raw_df):
    """add_zscore_features should add _zscore columns."""
    cleaned = preprocess(sample_raw_df)
    pivoted = pivot_parameters(cleaned)
    result = add_zscore_features(pivoted)
    zscore_cols = [c for c in result.columns if c.endswith("_zscore")]
    assert len(zscore_cols) > 0
    assert isinstance(result, pd.DataFrame)
