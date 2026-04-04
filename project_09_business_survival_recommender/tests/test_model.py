"""Tests for project_09 model module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import (
    SurvivalAnalyzer,
    train_random_forest,
    recommend_locations,
    _compute_metrics,
)


@pytest.fixture
def business_df():
    """Create a synthetic preprocessed business licences DataFrame."""
    np.random.seed(42)
    n = 300
    communities = ["BELTLINE", "DOWNTOWN", "KENSINGTON", "BRIDGELAND"]
    categories = ["Food Services", "Retail", "Professional Services"]
    df = pd.DataFrame({
        "getbusid": [f"B{i:04d}" for i in range(n)],
        "comdistnm": np.random.choice(communities, n),
        "business_category": np.random.choice(categories, n),
        "business_age_days": np.random.randint(30, 3000, n),
        "is_home_occupation": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "survived": np.random.choice([0, 1], n, p=[0.3, 0.7]),
        "business_count": np.random.randint(10, 500, n),
        "business_diversity": np.random.randint(1, 20, n),
        "avg_business_age": np.random.uniform(100, 2000, n),
        "issue_year": np.random.randint(2015, 2024, n),
        "issue_month": np.random.randint(1, 13, n),
        "homeoccind": np.random.choice(["Y", "N"], n),
        "licencetypes": np.random.choice(
            ["Food Services - Restaurant", "Retail - General"], n
        ),
    })
    # event_observed: 1 means closed (inverse of survived for KM)
    df["event_observed"] = 1 - df["survived"]
    return df


def test_survival_analyzer_kaplan_meier(business_df):
    """SurvivalAnalyzer should fit a KM curve and return a DataFrame."""
    sa = SurvivalAnalyzer()
    curve = sa.get_kaplan_meier_curve(
        business_df["business_age_days"],
        business_df["event_observed"],
    )
    assert isinstance(curve, pd.DataFrame)
    assert "survival_probability" in curve.columns
    assert len(curve) > 0


def test_train_random_forest_metrics(business_df):
    """train_random_forest should return a result dict with valid metrics."""
    result = train_random_forest(business_df)
    assert "metrics" in result
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert np.isfinite(result["metrics"]["f1"])


def test_recommend_locations(business_df):
    """recommend_locations should return a ranked DataFrame for a given type."""
    recs = recommend_locations(business_df, business_type="Food Services", top_n=3)
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) <= 3
    if len(recs) > 0:
        assert "overall_score" in recs.columns
