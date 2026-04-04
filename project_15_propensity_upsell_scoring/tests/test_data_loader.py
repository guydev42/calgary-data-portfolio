"""Tests for propensity upsell scoring data loader."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import engineer_features, get_feature_columns


@pytest.fixture
def sample_marketing_df():
    """Create a minimal marketing campaign dataframe."""
    n = 100
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "age": rng.randint(18, 70, n),
        "income": rng.uniform(30000, 150000, n).round(0),
        "tenure_months": rng.randint(1, 60, n),
        "monthly_spend": rng.uniform(20, 120, n).round(2),
        "data_usage_gb": rng.uniform(0.5, 50, n).round(1),
        "call_minutes": rng.uniform(10, 500, n).round(0),
        "sms_count": rng.randint(0, 200, n),
        "has_streaming": rng.choice([0, 1], n),
        "has_international": rng.choice([0, 1], n),
        "has_device_insurance": rng.choice([0, 1], n),
        "previous_upsell_response": rng.choice([0, 1], n),
        "current_plan": rng.choice(["Basic", "Standard", "Premium"], n),
        "channel_preference": rng.choice(["Email", "SMS", "App notification", "Direct mail"], n),
        "responded": rng.choice([0, 1], n, p=[0.85, 0.15]),
    })


def test_engineer_features_adds_columns(sample_marketing_df):
    """engineer_features should add derived columns."""
    result = engineer_features(sample_marketing_df)
    assert isinstance(result, pd.DataFrame)
    assert "revenue_per_tenure" in result.columns
    assert "usage_intensity" in result.columns
    assert "service_count" in result.columns
    assert "upsell_headroom" in result.columns
    assert "plan_encoded" in result.columns


def test_get_feature_columns_returns_list():
    """get_feature_columns should return a non-empty list of strings."""
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) > 0
    assert all(isinstance(c, str) for c in cols)


def test_engineer_features_no_nans_in_key_cols(sample_marketing_df):
    """Derived features should not introduce unexpected NaNs."""
    result = engineer_features(sample_marketing_df)
    for col in ["revenue_per_tenure", "service_count", "upsell_headroom"]:
        assert result[col].isna().sum() == 0
