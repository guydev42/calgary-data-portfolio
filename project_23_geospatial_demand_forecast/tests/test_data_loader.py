"""Tests for project_23 data_loader module."""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_demand_data, generate_zones_data, CALGARY_ZONES


def test_generate_demand_data_returns_dataframe():
    """generate_demand_data should return a DataFrame with expected columns."""
    df = generate_demand_data(n_samples=200, random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert "demand_count" in df.columns
    assert "zone_id" in df.columns
    assert "latitude" in df.columns


def test_generate_demand_data_demand_nonnegative():
    """demand_count should be non-negative."""
    df = generate_demand_data(n_samples=500, random_state=0)
    assert (df["demand_count"] >= 0).all()


def test_generate_zones_data():
    """generate_zones_data should return one row per Calgary zone."""
    zones = generate_zones_data()
    assert isinstance(zones, pd.DataFrame)
    assert len(zones) == len(CALGARY_ZONES)
    assert "zone_id" in zones.columns
    assert "distance_to_downtown_km" in zones.columns
