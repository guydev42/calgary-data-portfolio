"""Tests for project_26 benchmark module."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_data, NUMERIC_KPIS
from src.benchmark import (
    compute_percentiles,
    compute_all_percentiles,
    industry_summary,
    peer_comparison,
    gap_analysis,
    cross_industry_ranking,
)


def _make_synthetic_csv(tmp_path):
    """Create a minimal synthetic benchmark CSV for testing."""
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({
        "company_id": [f"C{i:03d}" for i in range(n)],
        "industry": rng.choice(["Tech", "Finance", "Healthcare"], n),
        "company_size": rng.choice(["Small", "Medium", "Large"], n),
        "region": rng.choice(["North America", "Europe", "Asia"], n),
        "avg_recognition_frequency": rng.uniform(0.5, 5.0, n).round(2),
        "avg_reward_value": rng.uniform(10, 500, n).round(2),
        "budget_per_employee": rng.uniform(50, 2000, n).round(2),
        "turnover_rate": rng.uniform(0.05, 0.40, n).round(3),
        "engagement_score": rng.uniform(3, 9, n).round(1),
        "eNPS": rng.randint(-20, 80, n),
        "training_hours_per_employee": rng.uniform(5, 80, n).round(1),
        "promotion_rate": rng.uniform(0.02, 0.20, n).round(3),
        "diversity_index": rng.uniform(0.3, 0.9, n).round(2),
        "revenue_per_employee": rng.uniform(50000, 500000, n).round(0),
        "profit_margin": rng.uniform(0.01, 0.35, n).round(3),
    })
    path = os.path.join(str(tmp_path), "industry_benchmark.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_df(tmp_path):
    csv_path = _make_synthetic_csv(tmp_path)
    return load_data(csv_path)


def test_compute_percentiles(sample_df):
    """compute_percentiles should return a Series of values between 0 and 100."""
    pctls = compute_percentiles(sample_df, "engagement_score")
    assert isinstance(pctls, pd.Series)
    assert pctls.min() >= 0
    assert pctls.max() <= 100


def test_industry_summary(sample_df):
    """industry_summary should return a DataFrame with mean, median, etc."""
    summary = industry_summary(sample_df)
    assert isinstance(summary, pd.DataFrame)
    assert "industry" in summary.columns
    assert "kpi" in summary.columns
    assert "mean" in summary.columns
    assert "median" in summary.columns
    assert len(summary) > 0


def test_peer_comparison(sample_df):
    """peer_comparison should return comparison for a valid company."""
    company_id = sample_df["company_id"].iloc[0]
    result = peer_comparison(sample_df, company_id)
    assert isinstance(result, pd.DataFrame)
    assert "KPI" in result.columns
    assert "Your value" in result.columns
    assert "Peer median" in result.columns
    assert len(result) == len(NUMERIC_KPIS)


def test_cross_industry_ranking(sample_df):
    """cross_industry_ranking should rank industries by a KPI."""
    ranking = cross_industry_ranking(sample_df, "engagement_score")
    assert isinstance(ranking, pd.DataFrame)
    assert "rank" in ranking.columns
    assert len(ranking) == sample_df["industry"].nunique()
