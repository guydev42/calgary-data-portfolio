"""Tests for property assessment valuator data loader."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import _generate_sample_data, preprocess_data, engineer_features


def test_generate_sample_data_returns_dataframe():
    """_generate_sample_data should return a DataFrame with expected columns."""
    df = _generate_sample_data(n=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "assessed_value" in df.columns
    assert "community" in df.columns
    assert "property_class" in df.columns


def test_preprocess_data_cleans_values():
    """preprocess_data should remove rows with zero/negative values and outliers."""
    df = _generate_sample_data(n=500)
    result = preprocess_data(df)
    assert isinstance(result, pd.DataFrame)
    assert (result["assessed_value"] > 0).all()
    assert len(result) <= len(df)


def test_engineer_features_adds_derived_columns():
    """engineer_features should add log_value and community aggregate columns."""
    df = _generate_sample_data(n=200)
    cleaned = preprocess_data(df)
    result = engineer_features(cleaned)
    assert isinstance(result, pd.DataFrame)
    assert "log_value" in result.columns
    assert "community_avg_value" in result.columns
    assert "community_median_value" in result.columns
    assert "land_use_frequency" in result.columns
