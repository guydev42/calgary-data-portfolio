"""Tests for transit ridership optimizer model."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import preprocess_ridership, engineer_features
from src.model import prepare_model_data, build_transit_network, get_network_stats


@pytest.fixture
def sample_ridership_engineered():
    """Create an engineered ridership dataframe."""
    df = pd.DataFrame({
        "year": list(range(2018, 2024)) * 12,
        "month": sorted(list(range(1, 13)) * 6),
        "ridership": np.random.RandomState(42).randint(80000, 160000, 72),
    })
    df = preprocess_ridership(df)
    df = engineer_features(df)
    return df


@pytest.fixture
def sample_stops_for_network():
    """Create stops data for network building."""
    return pd.DataFrame({
        "stop_id": [1, 2, 3, 4, 5],
        "latitude": [51.04, 51.05, 51.06, 51.07, 51.08],
        "longitude": [-114.06, -114.07, -114.08, -114.09, -114.10],
        "route_name": ["R1", "R1", "R1", "R2", "R2"],
        "stop_name": ["A", "B", "C", "D", "E"],
    })


def test_prepare_model_data_returns_valid_splits(sample_ridership_engineered):
    """prepare_model_data should return X, y with matching lengths."""
    X, y, _, features = prepare_model_data(sample_ridership_engineered)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert len(features) > 0


def test_build_transit_network_creates_graph(sample_stops_for_network):
    """build_transit_network should return a networkx Graph."""
    import networkx as nx
    G = build_transit_network(sample_stops_for_network)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() > 0


def test_get_network_stats_returns_dict(sample_stops_for_network):
    """get_network_stats should return a dict with expected keys."""
    G = build_transit_network(sample_stops_for_network)
    stats = get_network_stats(G)
    assert isinstance(stats, dict)
    assert "node_count" in stats
    assert "edge_count" in stats
    assert stats["node_count"] == 5
