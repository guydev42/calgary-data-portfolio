"""Tests for project_20 data_loader module."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import clean_text


def test_clean_text_returns_string():
    """clean_text should return a lowercase, cleaned string."""
    result = clean_text("This is a GREAT product! I love it 100%.")
    assert isinstance(result, str)
    # Should be lowercase with no punctuation or digits
    assert result == result.lower()
    assert "!" not in result
    assert "100" not in result


def test_clean_text_removes_stopwords():
    """clean_text should remove common English stopwords."""
    result = clean_text("The cat is on the mat")
    # 'the', 'is', 'on' are stopwords
    assert "the" not in result.split()
    assert "is" not in result.split()


def test_clean_text_empty_string():
    """clean_text should handle empty-ish input gracefully."""
    result = clean_text("a")
    assert isinstance(result, str)
