"""Tests for project_06 data_loader module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import (
    _build_census_features,
    _build_crime_features,
    FEATURE_COLUMNS,
)


@pytest.fixture
def sample_census_df():
    """Create a synthetic civic census DataFrame."""
    return pd.DataFrame({
        "code": ["BELTLINE", "BELTLINE", "DOWNTOWN", "DOWNTOWN"],
        "year": [2023, 2023, 2023, 2023],
        "age_range": ["20-24", "25-29", "20-24", "25-29"],
        "males": [1000, 1200, 800, 900],
        "females": [1100, 1300, 850, 950],
    })


@pytest.fixture
def sample_crime_df():
    """Create a synthetic crime statistics DataFrame."""
    return pd.DataFrame({
        "community": ["BELTLINE", "DOWNTOWN", "BELTLINE"],
        "category": ["Theft", "Assault", "Robbery"],
        "crime_count": [200, 150, 80],
    })


def test_build_census_features(sample_census_df):
    """_build_census_features should return community-level population stats."""
    result = _build_census_features(sample_census_df)
    assert isinstance(result, pd.DataFrame)
    assert "community" in result.columns
    assert "total_population" in result.columns


def test_build_crime_features(sample_crime_df):
    """_build_crime_features should aggregate crimes per community."""
    result = _build_crime_features(sample_crime_df)
    assert isinstance(result, pd.DataFrame)
    assert "total_crimes" in result.columns
    assert len(result) == 2  # two communities


def test_feature_columns_defined():
    """FEATURE_COLUMNS should be a non-empty list of strings."""
    assert isinstance(FEATURE_COLUMNS, list)
    assert len(FEATURE_COLUMNS) > 0
    assert all(isinstance(c, str) for c in FEATURE_COLUMNS)
