"""Tests for propensity upsell scoring model."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import build_models, decile_analysis, campaign_roi


def test_build_models_returns_dict():
    """build_models should return a dict of sklearn model instances."""
    models = build_models()
    assert isinstance(models, dict)
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    for name, model in models.items():
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_decile_analysis_returns_dataframe():
    """decile_analysis should return a DataFrame with expected columns."""
    rng = np.random.RandomState(42)
    y_test = rng.choice([0, 1], 1000, p=[0.85, 0.15])
    y_prob = rng.uniform(0, 1, 1000)
    result = decile_analysis(y_test, y_prob, n_deciles=10)
    assert isinstance(result, pd.DataFrame)
    assert "n_customers" in result.columns
    assert "response_rate" in result.columns
    assert "cumulative_response_pct" in result.columns
    assert "expected_revenue" in result.columns


def test_campaign_roi_returns_dict():
    """campaign_roi should return a dict with ROI metrics."""
    rng = np.random.RandomState(42)
    y_test = rng.choice([0, 1], 1000, p=[0.85, 0.15])
    y_prob = rng.uniform(0, 1, 1000)
    decile_df = decile_analysis(y_test, y_prob)
    result = campaign_roi(decile_df)
    assert isinstance(result, dict)
    assert "mass_roi_pct" in result
    assert "targeted_roi_pct" in result
    assert "cost_savings" in result
