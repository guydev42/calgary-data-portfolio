"""Tests for RAG document QA retrieval models."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import (
    TfidfRetriever,
    TermOverlapReranker,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


@pytest.fixture
def sample_chunks_and_metadata():
    """Create sample chunks and metadata for retriever tests."""
    chunks = [
        "The water treatment policy requires regular testing of pH levels.",
        "Calgary transit operates bus and train services across the city.",
        "Property taxes are assessed based on market value of real estate.",
        "Noise bylaws restrict construction activity during nighttime hours.",
        "Water quality monitoring includes testing for bacteria and chemicals.",
    ]
    metadata = [
        {"doc_id": "d1", "title": "Water Policy", "chunk_idx": 0},
        {"doc_id": "d2", "title": "Transit Guide", "chunk_idx": 0},
        {"doc_id": "d3", "title": "Tax Policy", "chunk_idx": 0},
        {"doc_id": "d4", "title": "Bylaws", "chunk_idx": 0},
        {"doc_id": "d1", "title": "Water Policy", "chunk_idx": 1},
    ]
    return chunks, metadata


def test_tfidf_retriever_fit_and_retrieve(sample_chunks_and_metadata):
    """TfidfRetriever should fit and retrieve relevant chunks."""
    chunks, metadata = sample_chunks_and_metadata
    retriever = TfidfRetriever()
    retriever.fit(chunks, metadata)
    results = retriever.retrieve("water quality testing", k=3)
    assert isinstance(results, list)
    assert len(results) == 3
    assert all("chunk" in r for r in results)
    assert all("score" in r for r in results)
    # Top result should be about water
    assert "water" in results[0]["chunk"].lower()


def test_term_overlap_reranker_scores():
    """TermOverlapReranker.score should return a float between 0 and 1."""
    reranker = TermOverlapReranker()
    score = reranker.score("water quality", "The water quality is good.")
    assert isinstance(score, float)
    assert score > 0
    zero_score = reranker.score("unrelated query", "completely different text")
    assert isinstance(zero_score, float)


def test_precision_and_recall_at_k():
    """precision_at_k and recall_at_k should compute correct values."""
    retrieved = ["d1", "d3", "d2", "d4"]
    relevant = ["d1", "d2"]

    p1 = precision_at_k(retrieved, relevant, k=1)
    assert p1 == 1.0  # d1 is relevant

    p2 = precision_at_k(retrieved, relevant, k=2)
    assert p2 == 0.5  # d1 relevant, d3 not

    r2 = recall_at_k(retrieved, relevant, k=2)
    assert r2 == 0.5  # found 1 of 2 relevant

    rr = reciprocal_rank(retrieved, relevant)
    assert rr == 1.0  # first result is relevant
