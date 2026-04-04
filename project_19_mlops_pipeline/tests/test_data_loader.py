"""Tests for project_19 data_loader module."""

import os
import sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import (
    generate_synthetic_churn_data,
    validate_schema,
    validate_nulls,
    ValidationResult,
)


def test_generate_synthetic_churn_data_returns_dataframe():
    """generate_synthetic_churn_data should return a DataFrame with expected columns."""
    df = generate_synthetic_churn_data(n_samples=100, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert "Churn" in df.columns
    assert "tenure" in df.columns


def test_generate_synthetic_churn_data_drift():
    """Drift flag should produce valid data without errors."""
    df = generate_synthetic_churn_data(n_samples=50, drift=True, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50


def test_validate_schema_on_synthetic_data():
    """validate_schema should return a ValidationResult with check entries."""
    df = generate_synthetic_churn_data(n_samples=50, seed=0)
    result = validate_schema(df)
    assert isinstance(result, ValidationResult)
    assert len(result.checks) > 0
    # Column presence checks should pass (synthetic data has all expected columns)
    col_check = next(c for c in result.checks if c["check"] == "columns_present")
    assert col_check["passed"] is True


def test_validate_nulls_on_synthetic_data():
    """validate_nulls should pass on synthetic data with no nulls."""
    df = generate_synthetic_churn_data(n_samples=50, seed=0)
    result = validate_nulls(df)
    assert isinstance(result, ValidationResult)
    assert result.passed is True
