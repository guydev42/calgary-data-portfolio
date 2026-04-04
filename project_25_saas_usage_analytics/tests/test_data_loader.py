"""Tests for project_25 data_loader module."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_synthetic_csv(tmp_path):
    """Create a minimal synthetic SaaS usage CSV for testing."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "user_id": range(1, n + 1),
        "company_id": rng.randint(1, 20, n),
        "plan_tier": rng.choice(["Free", "Pro", "Enterprise"], n),
        "industry": rng.choice(["SaaS", "Finance", "Healthcare"], n),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="3D").strftime("%Y-%m-%d"),
        "last_active_date": pd.date_range("2025-06-01", periods=n, freq="1D").strftime("%Y-%m-%d"),
        "daily_logins": rng.randint(0, 20, n),
        "session_duration_min": rng.uniform(1, 60, n).round(1),
        "features_used": rng.randint(1, 25, n),
        "actions_per_session": rng.randint(5, 100, n),
        "monthly_active_days": rng.randint(1, 30, n),
        "support_tickets": rng.randint(0, 10, n),
        "is_churned": rng.choice([0, 1], n, p=[0.75, 0.25]),
    })
    path = os.path.join(str(tmp_path), "saas_usage.csv")
    df.to_csv(path, index=False)
    return path


def test_load_and_prepare_returns_expected_types(tmp_path):
    """load_and_prepare should return arrays and feature names list."""
    from src.data_loader import load_and_prepare
    csv_path = _make_synthetic_csv(tmp_path)
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare(
        filepath=csv_path, test_size=0.2, random_state=42
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(feature_names, list)
    assert len(feature_names) == X_train.shape[1]


def test_load_and_prepare_feature_engineering(tmp_path):
    """Engineered features like engagement_score should be in feature names."""
    from src.data_loader import load_and_prepare
    csv_path = _make_synthetic_csv(tmp_path)
    _, _, _, _, feature_names = load_and_prepare(
        filepath=csv_path, test_size=0.2, random_state=42
    )
    assert any("engagement_score" in f for f in feature_names)
    assert any("feature_adoption_rate" in f for f in feature_names)
    assert any("account_age_days" in f for f in feature_names)


def test_load_and_prepare_target_is_binary(tmp_path):
    """Target values should be 0 or 1."""
    from src.data_loader import load_and_prepare
    csv_path = _make_synthetic_csv(tmp_path)
    _, _, y_train, y_test, _ = load_and_prepare(
        filepath=csv_path, test_size=0.2, random_state=42
    )
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})
