"""Tests for project_23 model module."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_demand_data
from src.model import engineer_features, _get_models, _mape


def test_engineer_features_adds_columns():
    """engineer_features should add spatial and temporal columns."""
    df = generate_demand_data(n_samples=200, random_state=0)
    df_feat, feature_cols, kmeans = engineer_features(df)
    assert "distance_to_downtown" in df_feat.columns
    assert "zone_cluster" in df_feat.columns
    assert "hour_sin" in df_feat.columns
    assert "is_rush_hour" in df_feat.columns
    assert isinstance(feature_cols, list)
    assert len(feature_cols) > 10


def test_get_models_returns_dict():
    """_get_models should return at least Ridge and Random Forest."""
    models = _get_models()
    assert isinstance(models, dict)
    assert "Ridge" in models
    assert "Random Forest" in models


def test_mape_calculation():
    """_mape should return a non-negative percentage."""
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])
    result = _mape(y_true, y_pred)
    assert isinstance(result, float)
    assert result >= 0
    # With zeros in y_true, should still work
    y_true_z = np.array([0, 10, 20])
    y_pred_z = np.array([1, 12, 18])
    result_z = _mape(y_true_z, y_pred_z)
    assert result_z >= 0
