"""Tests for fraud detection data loader."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_fraud_data


def test_generate_fraud_data_returns_dataframe():
    """generate_fraud_data should return a DataFrame with expected columns."""
    df = generate_fraud_data(n_samples=500)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
    assert "is_fraud" in df.columns
    assert "amount" in df.columns
    assert "time_hour" in df.columns


def test_generate_fraud_data_respects_fraud_rate():
    """Fraud rate should be approximately as specified."""
    df = generate_fraud_data(n_samples=10000, fraud_rate=0.02)
    actual_rate = df["is_fraud"].mean()
    assert 0.01 <= actual_rate <= 0.04  # within reasonable range


def test_generate_fraud_data_feature_ranges():
    """Feature values should be within expected ranges."""
    df = generate_fraud_data(n_samples=1000)
    assert (df["amount"] > 0).all()
    assert (df["time_hour"] >= 0).all() and (df["time_hour"] <= 23).all()
    assert set(df["is_fraud"].unique()).issubset({0, 1})
    assert (df["is_weekend"].isin([0, 1])).all()
    assert (df["is_night"].isin([0, 1])).all()
