"""Tests for project_26 data_loader module."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import (
    load_data,
    get_industries,
    get_kpi_display_names,
    filter_peers,
    NUMERIC_KPIS,
    CATEGORY_COLS,
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


def test_load_data_returns_dataframe(tmp_path):
    """load_data should return a DataFrame with correct columns."""
    csv_path = _make_synthetic_csv(tmp_path)
    df = load_data(csv_path)
    assert isinstance(df, pd.DataFrame)
    for kpi in NUMERIC_KPIS:
        assert kpi in df.columns
    for col in CATEGORY_COLS:
        assert df[col].dtype.name == "category"


def test_get_industries(tmp_path):
    """get_industries should return a sorted list of unique industry names."""
    csv_path = _make_synthetic_csv(tmp_path)
    df = load_data(csv_path)
    industries = get_industries(df)
    assert isinstance(industries, list)
    assert industries == sorted(industries)
    assert len(industries) == 3


def test_filter_peers(tmp_path):
    """filter_peers should narrow down the dataset correctly."""
    csv_path = _make_synthetic_csv(tmp_path)
    df = load_data(csv_path)
    filtered = filter_peers(df, industry="Tech")
    assert isinstance(filtered, pd.DataFrame)
    assert (filtered["industry"] == "Tech").all()
    assert len(filtered) < len(df)


def test_get_kpi_display_names():
    """get_kpi_display_names should return a dict covering all numeric KPIs."""
    names = get_kpi_display_names()
    assert isinstance(names, dict)
    for kpi in NUMERIC_KPIS:
        assert kpi in names
