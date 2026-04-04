"""Tests for project_07 data_loader module."""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess, clean_text


@pytest.fixture
def sample_permits_df():
    """Create a synthetic development permits DataFrame."""
    return pd.DataFrame({
        "applieddate": ["2024-01-15", "2024-03-20", "2024-06-10"],
        "statuscurrent": ["Approved - Conditions", "Cancelled", "Approved"],
        "description": [
            "New single detached house",
            "<b>Demo</b> existing structure & rebuild",
            "Change of use to retail 123",
        ],
        "category": ["Residential", "Commercial", "Commercial"],
        "landusedistrict": ["R-C1", "C-COR1", "C-COR2"],
        "communityname": ["Beltline", "Downtown", "Kensington"],
        "quadrant": ["SW", "SW", "NW"],
        "permitteddiscretionary": ["Permitted", "Discretionary", "Permitted"],
        "latitude": [51.04, 51.05, 51.06],
        "longitude": [-114.07, -114.06, -114.08],
    })


def test_preprocess_returns_dataframe(sample_permits_df):
    """preprocess should return a DataFrame with the approved column."""
    result = preprocess(sample_permits_df)
    assert isinstance(result, pd.DataFrame)
    assert "approved" in result.columns


def test_preprocess_creates_target(sample_permits_df):
    """preprocess should correctly set approved=1 for approved statuses."""
    result = preprocess(sample_permits_df)
    assert result["approved"].iloc[0] == 1  # "Approved - Conditions"
    assert result["approved"].iloc[1] == 0  # "Cancelled"
    assert result["approved"].iloc[2] == 1  # "Approved"


def test_clean_text():
    """clean_text should lowercase, strip HTML, and remove non-alpha chars."""
    result = clean_text("<b>Hello</b> World! 123")
    assert "<b>" not in result
    assert "123" not in result
    assert "hello" in result
