"""Tests for customer churn prediction data loader."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_churn_csv(tmp_path):
    """Create a minimal churn CSV file for testing."""
    n = 100
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "partner": rng.choice(["Yes", "No"], n),
        "dependents": rng.choice(["Yes", "No"], n),
        "phone_service": rng.choice(["Yes", "No"], n),
        "paperless_billing": rng.choice(["Yes", "No"], n),
        "tenure_months": rng.randint(1, 72, n),
        "monthly_charges": rng.uniform(20, 100, n).round(2),
        "total_charges": rng.uniform(100, 5000, n).round(2),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "internet_service": rng.choice(["DSL", "Fiber optic", "No"], n),
        "multiple_lines": rng.choice(["Yes", "No", "No phone service"], n),
        "online_security": rng.choice(["Yes", "No", "No internet service"], n),
        "online_backup": rng.choice(["Yes", "No", "No internet service"], n),
        "device_protection": rng.choice(["Yes", "No", "No internet service"], n),
        "tech_support": rng.choice(["Yes", "No", "No internet service"], n),
        "streaming_tv": rng.choice(["Yes", "No", "No internet service"], n),
        "streaming_movies": rng.choice(["Yes", "No", "No internet service"], n),
        "payment_method": rng.choice(["Electronic check", "Mailed check",
                                       "Bank transfer", "Credit card"], n),
        "churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })
    csv_path = tmp_path / "telco_churn.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_and_prepare_returns_arrays(sample_churn_csv):
    """load_and_prepare should return train/test splits and feature names."""
    from src.data_loader import load_and_prepare
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare(
        filepath=sample_churn_csv
    )
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(feature_names, list)
    assert len(feature_names) == X_train.shape[1]


def test_load_and_prepare_target_is_binary(sample_churn_csv):
    """Target variable should be binary (0 or 1)."""
    from src.data_loader import load_and_prepare
    _, _, y_train, y_test, _ = load_and_prepare(filepath=sample_churn_csv)
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_load_and_prepare_train_test_sizes(sample_churn_csv):
    """Train set should be ~80% and test set ~20%."""
    from src.data_loader import load_and_prepare
    X_train, X_test, _, _, _ = load_and_prepare(filepath=sample_churn_csv)
    total = X_train.shape[0] + X_test.shape[0]
    assert 0.15 <= X_test.shape[0] / total <= 0.25
