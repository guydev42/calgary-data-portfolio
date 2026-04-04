"""Tests for project_22 model module."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import build_user_item_matrix
from src.model import (
    user_based_cf,
    item_based_cf,
    train_svd,
    svd_recommend,
    rmse,
    mae,
    precision_at_k,
    ndcg_at_k,
)


@pytest.fixture
def small_matrix():
    """Build a small user-item matrix for testing."""
    ratings = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        "item_id": [10, 20, 30, 10, 40, 20, 30, 40, 10, 20],
        "rating":  [5,  3,  4,  4,  2,  5,  4,  3,  1,  5],
    })
    matrix, u2i, i2i, i2u, i2item = build_user_item_matrix(ratings)
    return matrix


def test_user_based_cf_returns_list(small_matrix):
    """user_based_cf should return a list of (item_idx, score) tuples."""
    recs = user_based_cf(small_matrix, user_idx=0, n_neighbors=3, top_n=5)
    assert isinstance(recs, list)
    for item in recs:
        assert len(item) == 2


def test_train_svd_shapes(small_matrix):
    """train_svd should return factors of consistent shape."""
    U, sigma, Vt, predicted, user_means = train_svd(small_matrix, n_factors=2)
    assert U.shape[0] == small_matrix.shape[0]
    assert Vt.shape[1] == small_matrix.shape[1]
    assert predicted.shape == small_matrix.shape
    assert len(sigma) == 2


def test_evaluation_metrics():
    """rmse, mae, precision_at_k, ndcg_at_k should return correct types."""
    actual = np.array([3.0, 4.0, 5.0])
    predicted = np.array([2.5, 4.5, 4.0])
    assert isinstance(rmse(actual, predicted), float)
    assert isinstance(mae(actual, predicted), float)
    assert rmse(actual, predicted) >= 0
    assert mae(actual, predicted) >= 0

    rec = [1, 2, 3, 4, 5]
    rel = [2, 5, 7]
    assert 0 <= precision_at_k(rec, rel, 5) <= 1
    assert 0 <= ndcg_at_k(rec, rel, 5) <= 1
