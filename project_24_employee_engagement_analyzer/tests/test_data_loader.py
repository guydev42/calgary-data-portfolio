"""Tests for project_24 data_loader module."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_synthetic_csv(tmp_path):
    """Create a minimal synthetic employee engagement CSV for testing."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "employee_id": range(1, n + 1),
        "department": rng.choice(["Engineering", "Sales", "HR"], n),
        "role_level": rng.choice(["Junior", "Mid", "Senior"], n),
        "tenure_months": rng.randint(1, 120, n),
        "satisfaction_survey": rng.uniform(1, 10, n).round(1),
        "recognition_events_given": rng.randint(0, 15, n),
        "recognition_events_received": rng.randint(0, 15, n),
        "avg_reward_value": rng.uniform(0, 200, n).round(2),
        "reward_type": rng.choice(["Gift Card", "Cash", "Points"], n),
        "absenteeism_days": rng.randint(0, 30, n),
        "engagement_score": rng.uniform(1, 10, n).round(1),
        "is_disengaged": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })
    path = os.path.join(str(tmp_path), "employee_engagement.csv")
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


def test_load_and_prepare_stratified_split(tmp_path):
    """Train and test sets should both contain both classes."""
    from src.data_loader import load_and_prepare
    csv_path = _make_synthetic_csv(tmp_path)
    X_train, X_test, y_train, y_test, _ = load_and_prepare(
        filepath=csv_path, test_size=0.2, random_state=42
    )
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})


def test_load_and_prepare_feature_engineering(tmp_path):
    """Engineered features like recognition_balance should be present."""
    from src.data_loader import load_and_prepare
    csv_path = _make_synthetic_csv(tmp_path)
    _, _, _, _, feature_names = load_and_prepare(
        filepath=csv_path, test_size=0.2, random_state=42
    )
    assert any("recognition_balance" in f for f in feature_names)
    assert any("absenteeism_rate" in f for f in feature_names)
