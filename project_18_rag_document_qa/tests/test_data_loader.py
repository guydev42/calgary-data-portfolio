"""Tests for RAG document QA data loader."""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import chunk_text, build_chunk_index, preprocess_text


def test_chunk_text_basic():
    """chunk_text should split text into chunks of specified size."""
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_empty_input():
    """chunk_text should return empty list for empty input."""
    assert chunk_text("", chunk_size=100, overlap=10) == []
    assert chunk_text("hello", chunk_size=0, overlap=0) == []


def test_build_chunk_index_returns_parallel_lists():
    """build_chunk_index should return chunks and metadata of same length."""
    documents = [
        {"doc_id": "d1", "title": "Doc One", "text": "This is the first document. " * 20},
        {"doc_id": "d2", "title": "Doc Two", "text": "This is the second document. " * 20},
    ]
    chunks, metadata = build_chunk_index(documents, chunk_size=100, overlap=20)
    assert isinstance(chunks, list)
    assert isinstance(metadata, list)
    assert len(chunks) == len(metadata)
    assert len(chunks) > 0
    assert all("doc_id" in m for m in metadata)
    assert all("title" in m for m in metadata)


def test_preprocess_text_lowercases_and_strips():
    """preprocess_text should lowercase and remove special characters."""
    result = preprocess_text("Hello WORLD! @#$ Test 123.")
    assert isinstance(result, str)
    assert result == result.lower()
    assert "@" not in result
    assert "#" not in result
