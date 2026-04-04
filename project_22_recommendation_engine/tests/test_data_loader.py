"""Tests for project_22 data_loader module."""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import build_user_item_matrix, train_test_split_ratings


def _make_ratings():
    """Create a small synthetic ratings DataFrame."""
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
        "item_id": [10, 20, 10, 30, 20, 30, 40, 10, 40, 20],
        "rating": [5, 3, 4, 2, 5, 4, 3, 1, 5, 4],
    })


def test_build_user_item_matrix_returns_sparse():
    """build_user_item_matrix should return a sparse matrix and mapping dicts."""
    ratings = _make_ratings()
    matrix, u2i, i2i, i2u, i2item = build_user_item_matrix(ratings)
    assert isinstance(matrix, csr_matrix)
    assert matrix.shape[0] == ratings["user_id"].nunique()
    assert matrix.shape[1] == ratings["item_id"].nunique()
    assert matrix.nnz == len(ratings)


def test_build_user_item_matrix_mappings():
    """Mapping dicts should be consistent with the matrix dimensions."""
    ratings = _make_ratings()
    matrix, u2i, i2i, i2u, i2item = build_user_item_matrix(ratings)
    assert len(u2i) == matrix.shape[0]
    assert len(i2i) == matrix.shape[1]
    assert len(i2u) == matrix.shape[0]
    assert len(i2item) == matrix.shape[1]


def test_train_test_split_ratings():
    """train_test_split_ratings should split preserving at least one train per user."""
    ratings = _make_ratings()
    train, test = train_test_split_ratings(ratings, test_size=0.3, random_state=42)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(ratings)
    # Every user with >1 rating should appear in train
    for uid in ratings["user_id"].unique():
        assert uid in train["user_id"].values
