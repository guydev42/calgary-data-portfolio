"""Tests for project_05 model module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess, add_rolling_features
from model import encode_categorical, prepare_features_target, train_model


@pytest.fixture
def prepared_df():
    """Create a preprocessed shelter DataFrame ready for modeling."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    shelters = ["Alpha House", "Calgary Drop-In", "Mustard Seed"]
    rows = []
    for shelter in shelters:
        for d in dates:
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "shelter": shelter,
                "sheltertype": "Emergency",
                "organization": "Org A",
                "capacity": 100,
                "overnight": np.random.randint(50, 110),
            })
    df = pd.DataFrame(rows)
    df = preprocess(df)
    df = add_rolling_features(df)
    return df


def test_encode_categorical(prepared_df):
    """encode_categorical should add a sheltertype_encoded column."""
    df_enc, encoders = encode_categorical(prepared_df)
    assert "sheltertype_encoded" in df_enc.columns
    assert "sheltertype" in encoders


def test_prepare_features_target(prepared_df):
    """prepare_features_target should return X and y with matching lengths."""
    df_enc, _ = encode_categorical(prepared_df)
    X, y = prepare_features_target(df_enc)
    assert len(X) == len(y)
    assert len(X) > 0


def test_train_model_returns_metrics(prepared_df):
    """train_model should produce a result dict with valid test metrics."""
    result = train_model(prepared_df, model_name="random_forest")
    assert "test_metrics" in result
    assert np.isfinite(result["test_metrics"]["MAE"])
    assert np.isfinite(result["test_metrics"]["R2"])
