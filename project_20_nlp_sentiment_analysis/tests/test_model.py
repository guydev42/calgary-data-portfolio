"""Tests for project_20 model module."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import _get_models, train_and_evaluate
from sklearn.feature_extraction.text import TfidfVectorizer


def test_get_models_returns_dict():
    """_get_models should return a dict of named sklearn classifiers."""
    models = _get_models()
    assert isinstance(models, dict)
    assert len(models) >= 2
    assert "Logistic Regression" in models


def test_train_and_evaluate_runs():
    """train_and_evaluate should run on small synthetic text data without error."""
    # Need enough samples so TF-IDF min_df=3 keeps terms AND cv=5 works (5+ per class)
    pos = ["great product love amazing quality", "amazing fantastic wonderful product great",
           "love this great product amazing quality", "wonderful product love great fantastic",
           "best product ever love amazing great", "superb product quality great love"]
    neg = ["terrible awful waste product bad", "horrible bad product never again waste",
           "bad terrible product awful waste horrible", "worst product bad terrible never buy",
           "awful product waste terrible bad horrible", "dreadful product bad waste terrible"]
    neu = ["okay average nothing special product", "decent alright fine product average okay",
           "average product okay decent nothing special", "fine product okay average alright",
           "mediocre product average okay decent fine", "standard product okay average decent"]
    texts_train = np.array(pos + neg + neu)
    y_train = np.array([2]*6 + [0]*6 + [1]*6)
    texts_test = np.array([
        "wonderful best product great ever",
        "worst purchase product bad regret",
        "okay product average decent",
    ])
    y_test = np.array([2, 0, 1])

    results = train_and_evaluate(texts_train, texts_test, y_train, y_test)
    assert isinstance(results, dict)
    assert len(results) >= 2
    for name, r in results.items():
        assert "accuracy" in r
        assert "macro_f1" in r
        assert 0 <= r["accuracy"] <= 1


def test_tfidf_vectorizer_integration():
    """TfidfVectorizer used by the model should produce expected shape."""
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=1)
    texts = ["good product", "bad product", "average item"]
    X = tfidf.fit_transform(texts)
    assert X.shape[0] == 3
    assert X.shape[1] > 0
