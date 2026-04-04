"""Tests for A/B test framework analysis functions (experiment.py)."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiment import frequentist_test, bayesian_ab, sequential_test


@pytest.fixture
def sample_ab_data():
    """Create synthetic A/B test data."""
    rng = np.random.RandomState(42)
    control = rng.binomial(1, 0.05, 1000)
    treatment = rng.binomial(1, 0.08, 1000)
    return control, treatment


def test_frequentist_test_conversion(sample_ab_data):
    """frequentist_test should return expected keys for conversion metric."""
    control, treatment = sample_ab_data
    result = frequentist_test(control, treatment, metric="conversion")
    assert isinstance(result, dict)
    assert "p_value" in result
    assert "effect" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "significant" in result
    assert isinstance(result["significant"], bool)


def test_bayesian_ab_returns_probabilities(sample_ab_data):
    """bayesian_ab should return probability that treatment is better."""
    control, treatment = sample_ab_data
    result = bayesian_ab(control, treatment)
    assert isinstance(result, dict)
    assert "prob_treatment_better" in result
    assert 0 <= result["prob_treatment_better"] <= 1
    assert "expected_lift" in result
    assert "control_posterior" in result
    assert len(result["control_posterior"]) == 100000


def test_sequential_test_returns_results():
    """sequential_test should return analysis at each interim look."""
    rng = np.random.RandomState(42)
    data = pd.DataFrame({
        "variant": ["control"] * 500 + ["treatment"] * 500,
        "converted": np.concatenate([
            rng.binomial(1, 0.05, 500),
            rng.binomial(1, 0.08, 500),
        ]),
    })
    result = sequential_test(data, n_looks=5)
    assert isinstance(result, dict)
    assert "results" in result
    assert len(result["results"]) == 5
    assert "stopped_early" in result
    assert isinstance(result["stopped_early"], bool)
