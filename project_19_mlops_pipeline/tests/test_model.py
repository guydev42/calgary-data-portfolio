"""Tests for project_19 model module."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_synthetic_churn_data
from src.model import (
    ModelTrainer,
    ModelEvaluator,
    FeatureEngineer,
    compute_psi,
    compare_models,
)


@pytest.fixture
def sample_data():
    return generate_synthetic_churn_data(n_samples=200, seed=0)


def test_feature_engineer_add_derived_features(sample_data):
    """FeatureEngineer.add_derived_features should add new columns."""
    fe = FeatureEngineer()
    df_feat = fe.add_derived_features(sample_data)
    assert "AvgMonthlySpend" in df_feat.columns
    assert "tenure_bucket" in df_feat.columns
    assert "num_services" in df_feat.columns
    assert len(df_feat) == len(sample_data)


def test_model_trainer_fit_and_predict(sample_data):
    """ModelTrainer should fit and produce predictions of the right shape."""
    trainer = ModelTrainer()
    trainer.fit(sample_data)
    preds = trainer.predict(sample_data)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(sample_data)
    assert set(preds).issubset({0, 1})


def test_compute_psi_returns_dict(sample_data):
    """compute_psi should return a dict of float PSI scores."""
    ref = sample_data.head(100)
    cur = sample_data.tail(100)
    result = compute_psi(ref, cur)
    assert isinstance(result, dict)
    for v in result.values():
        assert isinstance(v, float)


def test_compare_models_returns_dict():
    """compare_models should return comparison outcome dict."""
    a = {"roc_auc": 0.80, "f1": 0.60}
    b = {"roc_auc": 0.85, "f1": 0.65}
    result = compare_models(a, b)
    assert isinstance(result, dict)
    assert "promote_challenger" in result
    assert result["promote_challenger"] is True
