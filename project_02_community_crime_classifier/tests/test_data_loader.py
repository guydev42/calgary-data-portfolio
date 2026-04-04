"""Tests for project_02 data_loader module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess_crime_data, create_community_features


@pytest.fixture
def sample_crime_df():
    """Create a synthetic crime statistics DataFrame."""
    return pd.DataFrame({
        "community": ["BELTLINE", "DOWNTOWN", "BELTLINE", "DOWNTOWN"],
        "category": ["Theft", "Assault", "Theft", "Robbery"],
        "crime_count": [100, 50, 80, 30],
        "year": [2023, 2023, 2023, 2023],
        "month": [1, 1, 2, 2],
    })


@pytest.fixture
def sample_census_df():
    """Create a synthetic census DataFrame."""
    return pd.DataFrame({
        "code": ["BELTLINE", "DOWNTOWN"],
        "year": [2023, 2023],
        "males": [5000, 3000],
        "females": [5200, 3100],
    })


def test_preprocess_crime_returns_dataframe(sample_crime_df):
    """preprocess_crime_data should return a cleaned DataFrame."""
    result = preprocess_crime_data(sample_crime_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_preprocess_crime_types(sample_crime_df):
    """preprocess_crime_data should convert crime_count and year to int."""
    result = preprocess_crime_data(sample_crime_df)
    assert result["crime_count"].dtype in [np.int32, np.int64]
    assert result["year"].dtype in [np.int32, np.int64]


def test_create_community_features(sample_crime_df, sample_census_df):
    """create_community_features should merge crime and census data."""
    census = sample_census_df.copy()
    census["total_pop"] = census["males"] + census["females"]
    result = create_community_features(sample_crime_df, census)
    assert isinstance(result, pd.DataFrame)
    assert "total_crimes" in result.columns
