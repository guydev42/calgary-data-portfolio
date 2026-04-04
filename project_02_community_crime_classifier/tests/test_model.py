"""Tests for project_02 model module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import create_risk_labels, prepare_classification_data, train_classifiers


@pytest.fixture
def community_df():
    """Create a synthetic community-level DataFrame for classification."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        "community": [f"COMM_{i}" for i in range(n)],
        "total_crimes": np.random.randint(10, 1000, n),
        "avg_monthly_crimes": np.random.uniform(1, 50, n),
        "crime_categories": np.random.randint(1, 10, n),
        "years_of_data": np.random.randint(1, 5, n),
        "total_population": np.random.randint(500, 50000, n),
        "crime_rate_per_1000": np.random.uniform(1, 100, n),
    })


def test_create_risk_labels(community_df):
    """create_risk_labels should add a risk_level column with Low/Medium/High."""
    result = create_risk_labels(community_df)
    assert "risk_level" in result.columns
    assert set(result["risk_level"].unique()).issubset({"Low", "Medium", "High"})


def test_prepare_classification_data(community_df):
    """prepare_classification_data should return X, y, encoder, and feature names."""
    labeled = create_risk_labels(community_df)
    X, y, le, feature_cols = prepare_classification_data(labeled)
    assert len(X) == len(y)
    assert len(X) > 0
    assert len(feature_cols) > 0


def test_train_classifiers_returns_metrics(community_df):
    """train_classifiers should return trained models with valid accuracy scores."""
    labeled = create_risk_labels(community_df)
    X, y, le, _ = prepare_classification_data(labeled)
    trained_models, results, scaler, X_test, y_test = train_classifiers(X, y)
    assert len(results) > 0
    for name, metrics in results.items():
        assert 0 <= metrics["Accuracy"] <= 1
        assert 0 <= metrics["F1 (Weighted)"] <= 1
