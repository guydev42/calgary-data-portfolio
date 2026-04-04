"""Tests for project_09 data_loader module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import (
    _parse_dates,
    _create_survived_column,
    _extract_business_category,
    _encode_home_occupation,
)


@pytest.fixture
def sample_licences_df():
    """Create a synthetic business licences DataFrame."""
    return pd.DataFrame({
        "getbusid": ["B001", "B002", "B003", "B004"],
        "tradename": ["Cafe A", "Shop B", "Restaurant C", "Tech D"],
        "homeoccind": ["N", "Y", "N", "N"],
        "address": ["100 1st St SW", "200 2nd St NW", "300 3rd Ave SE", "400 4th St NE"],
        "comdistcd": ["BEL", "DT", "BEL", "DT"],
        "comdistnm": ["BELTLINE", "DOWNTOWN", "BELTLINE", "DOWNTOWN"],
        "licencetypes": [
            "Food Services - Restaurant",
            "Retail - General",
            "Food Services - Cafe",
            "Professional Services - Technology",
        ],
        "first_iss_dt": ["2020-01-15", "2019-06-20", "2021-03-10", "2018-11-05"],
        "exp_dt": ["2025-01-15", None, "2024-03-10", "2023-11-05"],
        "jobstatusdesc": ["Licensed", "Cancelled", "Renewal Licensed", "Expired"],
    })


def test_parse_dates(sample_licences_df):
    """_parse_dates should create business_age_days and issue_year columns."""
    result = _parse_dates(sample_licences_df)
    assert "business_age_days" in result.columns
    assert "issue_year" in result.columns
    assert result["business_age_days"].min() >= 0


def test_create_survived_column(sample_licences_df):
    """_create_survived_column should produce a binary survived target."""
    dated = _parse_dates(sample_licences_df)
    result = _create_survived_column(dated)
    assert "survived" in result.columns
    assert set(result["survived"].unique()).issubset({0, 1})


def test_extract_business_category(sample_licences_df):
    """_extract_business_category should parse the first part of licencetypes."""
    result = _extract_business_category(sample_licences_df)
    assert "business_category" in result.columns
    assert result["business_category"].iloc[0] == "Food Services"
