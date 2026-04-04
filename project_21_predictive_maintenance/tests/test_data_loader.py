"""Tests for project_21 data_loader module."""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_sensor_data


def test_generate_sensor_data_returns_dataframe():
    """generate_sensor_data should return a DataFrame with expected columns."""
    df = generate_sensor_data(n_readings=200, n_machines=5, random_state=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 200
    assert "failure_within_7days" in df.columns
    assert "temperature" in df.columns
    assert "vibration" in df.columns


def test_generate_sensor_data_target_distribution():
    """Target column should contain only 0 and 1."""
    df = generate_sensor_data(n_readings=500, n_machines=10, random_state=0)
    assert set(df["failure_within_7days"].unique()).issubset({0, 1})


def test_generate_sensor_data_has_derived_features():
    """Generated data should include rolling and interaction features."""
    df = generate_sensor_data(n_readings=100, n_machines=5, random_state=0)
    assert "rolling_mean_temp_24h" in df.columns
    assert "rolling_std_vibration_24h" in df.columns
    assert "temp_pressure_ratio" in df.columns
