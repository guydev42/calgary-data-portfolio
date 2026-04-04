"""Tests for water quality anomaly detection models."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import (
    IsolationForestDetector,
    StatisticalDetector,
    ZScoreDetector,
    WaterQualityAnomalyDetector,
)


@pytest.fixture
def sample_features():
    """Create a synthetic feature matrix with a few outliers."""
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, size=(100, 4))
    # Inject outliers
    X[0] = [10, 10, 10, 10]
    X[1] = [-10, -10, -10, -10]
    return X


def test_isolation_forest_returns_binary(sample_features):
    """IsolationForestDetector.fit_predict should return 0/1 array."""
    detector = IsolationForestDetector(contamination=0.05)
    labels = detector.fit_predict(sample_features)
    assert isinstance(labels, np.ndarray)
    assert set(np.unique(labels)).issubset({0, 1})
    assert len(labels) == sample_features.shape[0]


def test_statistical_detector_flags_outliers(sample_features):
    """StatisticalDetector should flag extreme values as anomalies."""
    detector = StatisticalDetector(n_std=3.0)
    labels = detector.fit_predict(sample_features)
    assert isinstance(labels, np.ndarray)
    assert labels[0] == 1  # injected outlier should be flagged
    assert labels[1] == 1


def test_ensemble_fit_predict_returns_dict(sample_features):
    """WaterQualityAnomalyDetector.fit_predict should return expected keys."""
    detector = WaterQualityAnomalyDetector(contamination=0.1)
    results = detector.fit_predict(sample_features)
    assert isinstance(results, dict)
    expected_keys = {"isolation_forest", "lof", "statistical", "zscore",
                     "ensemble_score", "anomaly"}
    assert expected_keys.issubset(results.keys())
    assert results["anomaly"].dtype == int
    assert len(results["ensemble_score"]) == sample_features.shape[0]
