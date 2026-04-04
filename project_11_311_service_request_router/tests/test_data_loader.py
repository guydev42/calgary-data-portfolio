"""Tests for 311 service request router data loader."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import preprocess_data, engineer_features


@pytest.fixture
def sample_311_df():
    """Create a minimal 311 service request dataframe."""
    n = 50
    return pd.DataFrame({
        "created_date": pd.date_range("2023-01-01", periods=n, freq="h"),
        "closed_date": pd.date_range("2023-01-02", periods=n, freq="h"),
        "service_request_type": ["Pothole"] * 20 + ["Noise"] * 20 + ["Graffiti"] * 10,
        "status": ["Open"] * 25 + ["Closed"] * 25,
        "agency_responsible": ["Roads"] * 20 + ["Bylaw"] * 20 + ["Parks"] * 10,
        "ward": ["W01"] * 25 + ["W02"] * 25,
        "community": ["Beltline"] * 25 + ["Kensington"] * 25,
        "channel": ["Phone"] * 30 + ["Web"] * 20,
    })


def test_preprocess_returns_dataframe(sample_311_df):
    """preprocess_data should return a cleaned DataFrame."""
    result = preprocess_data(sample_311_df)
    assert isinstance(result, pd.DataFrame)
    assert "resolution_hours" in result.columns
    assert "year" in result.columns
    assert "hour" in result.columns


def test_preprocess_drops_missing_key_fields():
    """preprocess_data should drop rows missing agency_responsible."""
    df = pd.DataFrame({
        "created_date": ["2023-01-01"],
        "service_request_type": ["Pothole"],
        "agency_responsible": [None],
        "channel": ["Phone"],
        "ward": ["W01"],
        "community": ["Beltline"],
    })
    result = preprocess_data(df)
    assert len(result) == 0


def test_engineer_features_adds_columns(sample_311_df):
    """engineer_features should add community_request_count and service_type_frequency."""
    cleaned = preprocess_data(sample_311_df)
    result = engineer_features(cleaned)
    assert isinstance(result, pd.DataFrame)
    assert "community_request_count" in result.columns
    assert "service_type_frequency" in result.columns
