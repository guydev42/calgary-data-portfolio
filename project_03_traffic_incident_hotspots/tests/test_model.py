"""Tests for project_03 model module (clustering + classification)."""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import SpatialClusterAnalyzer, IncidentClassifier
from data_loader import preprocess_dataframe, create_classification_features


@pytest.fixture
def sample_coordinates():
    """Create synthetic lat/lon coordinates for clustering."""
    np.random.seed(42)
    n = 200
    lat = np.random.uniform(51.0, 51.1, n)
    lon = np.random.uniform(-114.2, -114.0, n)
    return np.column_stack([lat, lon])


@pytest.fixture
def classification_data():
    """Create synthetic traffic data suitable for classification."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "start_dt": pd.date_range("2023-01-01", periods=n, freq="h"),
        "quadrant": np.random.choice(["NW", "NE", "SW", "SE"], n),
        "latitude": np.random.uniform(50.9, 51.2, n),
        "longitude": np.random.uniform(-114.3, -113.9, n),
        "count": np.random.randint(1, 5, n),
    })
    processed = preprocess_dataframe(df)
    X, y = create_classification_features(processed)
    return X, y


def test_kmeans_clustering(sample_coordinates):
    """SpatialClusterAnalyzer.fit_kmeans should return labels of correct length."""
    analyzer = SpatialClusterAnalyzer()
    labels = analyzer.fit_kmeans(sample_coordinates, n_clusters=4)
    assert len(labels) == len(sample_coordinates)
    assert set(labels).issubset(set(range(4)))


def test_dbscan_clustering(sample_coordinates):
    """SpatialClusterAnalyzer.fit_dbscan should return integer labels."""
    analyzer = SpatialClusterAnalyzer()
    labels = analyzer.fit_dbscan(sample_coordinates, eps=0.01, min_samples=3)
    assert len(labels) == len(sample_coordinates)
    assert labels.dtype in [np.int32, np.int64]


def test_incident_classifier(classification_data):
    """IncidentClassifier should train and produce valid metrics."""
    X, y = classification_data
    if len(X) == 0:
        pytest.skip("Not enough data for classification test")
    classifier = IncidentClassifier()
    results = classifier.train_and_evaluate(X, y)
    assert len(results) > 0
    for name, metrics in results.items():
        assert 0 <= metrics["accuracy"] <= 1
