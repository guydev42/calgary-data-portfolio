"""Tests for project_01 data_loader module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess_data, engineer_features


@pytest.fixture
def sample_permits_df():
    """Create a small synthetic building permits DataFrame."""
    return pd.DataFrame({
        "permitnum": ["BP2024-0001", "BP2024-0002", "BP2024-0003"],
        "permittype": ["Building", "Electrical", "Building"],
        "permitclass": ["Residential", "Commercial", "Residential"],
        "permitclassgroup": ["House", "Commercial", "House"],
        "workclass": ["New", "Alteration", "New"],
        "workclassgroup": ["New Construction", "Alteration", "New Construction"],
        "statuscurrent": ["Issued", "Complete", "Issued"],
        "applieddate": ["2024-01-15", "2024-03-20", "2024-06-10"],
        "issueddate": ["2024-02-01", "2024-04-01", "2024-07-01"],
        "completeddate": [None, "2024-05-01", None],
        "description": ["New house", "Electrical upgrade", "New house"],
        "housingunits": [1, 0, 2],
        "estprojectcost": [250000, 15000, 500000],
        "totalsqft": [2000, 500, 3500],
        "communitycode": ["BEL", "DT", "BEL"],
        "communityname": ["Beltline", "Downtown", "Beltline"],
        "latitude": [51.04, 51.05, 51.04],
        "longitude": [-114.07, -114.06, -114.07],
    })


def test_preprocess_returns_dataframe(sample_permits_df):
    """preprocess_data should return a DataFrame."""
    result = preprocess_data(sample_permits_df)
    assert isinstance(result, pd.DataFrame)


def test_preprocess_creates_date_features(sample_permits_df):
    """preprocess_data should extract year, month, and dayofweek from applieddate."""
    result = preprocess_data(sample_permits_df)
    for col in ["apply_year", "apply_month", "apply_dayofweek"]:
        assert col in result.columns, f"Missing expected column: {col}"


def test_engineer_features_adds_log_cost(sample_permits_df):
    """engineer_features should create log_cost and community aggregate columns."""
    preprocessed = preprocess_data(sample_permits_df)
    result = engineer_features(preprocessed)
    assert "log_cost" in result.columns
    assert np.all(np.isfinite(result["log_cost"]))
