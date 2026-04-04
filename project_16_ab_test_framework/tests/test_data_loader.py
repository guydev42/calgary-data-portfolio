"""Tests for A/B test framework experiment module (data/analysis functions)."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiment import power_analysis, frequentist_test, multiple_comparison_correction


def test_power_analysis_returns_dict():
    """power_analysis should return a dict with sample size info."""
    result = power_analysis(baseline_rate=0.03, mde=0.01)
    assert isinstance(result, dict)
    assert "sample_size_per_group" in result
    assert "total_sample_size" in result
    assert result["sample_size_per_group"] > 0
    assert result["total_sample_size"] == 2 * result["sample_size_per_group"]


def test_power_analysis_larger_mde_needs_fewer_samples():
    """A larger MDE should require fewer samples."""
    small_mde = power_analysis(baseline_rate=0.05, mde=0.005)
    large_mde = power_analysis(baseline_rate=0.05, mde=0.02)
    assert small_mde["sample_size_per_group"] > large_mde["sample_size_per_group"]


def test_multiple_comparison_correction_bonferroni():
    """Bonferroni correction should multiply p-values by number of tests."""
    p_values = [0.01, 0.04, 0.06]
    result = multiple_comparison_correction(p_values, method="bonferroni")
    assert isinstance(result, dict)
    assert len(result["adjusted_p_values"]) == 3
    assert result["adjusted_p_values"][0] == pytest.approx(0.03, abs=1e-6)
    assert all(a >= o for a, o in zip(result["adjusted_p_values"], p_values))
