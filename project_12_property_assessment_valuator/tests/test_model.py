"""Tests for property assessment valuator model."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import _generate_sample_data, preprocess_data, engineer_features
from src.model import prepare_model_data, get_feature_importance


@pytest.fixture
def sample_engineered_df():
    """Create an engineered property assessment dataframe."""
    df = _generate_sample_data(n=300)
    df = preprocess_data(df)
    df = engineer_features(df)
    return df


def test_prepare_model_data_returns_correct_types(sample_engineered_df):
    """prepare_model_data should return X, y, encoders, feature_names."""
    X, y, encoders, features = prepare_model_data(sample_engineered_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(encoders, dict)
    assert isinstance(features, list)
    assert len(X) == len(y)
    assert len(X) > 0


def test_prepare_model_data_target_is_log_value(sample_engineered_df):
    """Target variable should be log_value (all positive)."""
    _, y, _, _ = prepare_model_data(sample_engineered_df)
    assert (y > 0).all()


def test_get_feature_importance_empty_for_non_tree():
    """get_feature_importance should return empty DataFrame for non-tree models."""
    from sklearn.linear_model import Ridge
    model = Ridge()
    result = get_feature_importance(model, ["a", "b"])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
