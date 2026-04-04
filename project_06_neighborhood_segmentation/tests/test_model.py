"""Tests for project_06 model module (clustering)."""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import train_kmeans, train_agglomerative, fit_pca, compute_elbow


@pytest.fixture
def scaled_features():
    """Create a synthetic scaled feature matrix."""
    np.random.seed(42)
    return np.random.randn(80, 5)


def test_train_kmeans(scaled_features):
    """train_kmeans should return a fitted KMeans with correct number of labels."""
    model = train_kmeans(scaled_features, n_clusters=3)
    assert hasattr(model, "labels_")
    assert len(model.labels_) == len(scaled_features)
    assert len(set(model.labels_)) == 3


def test_train_agglomerative(scaled_features):
    """train_agglomerative should return labels matching the input size."""
    model = train_agglomerative(scaled_features, n_clusters=4)
    assert hasattr(model, "labels_")
    assert len(model.labels_) == len(scaled_features)


def test_fit_pca(scaled_features):
    """fit_pca should return a PCA object and transformed data."""
    pca, X_pca = fit_pca(scaled_features, n_components=3)
    assert X_pca.shape == (len(scaled_features), 3)
    assert hasattr(pca, "explained_variance_ratio_")
    assert pca.explained_variance_ratio_.sum() <= 1.0
