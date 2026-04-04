"""Tests for project_07 model module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import preprocess
from model import FeatureBuilder, split_data, evaluate_model


@pytest.fixture
def preprocessed_df():
    """Create a preprocessed development permits DataFrame for modeling."""
    np.random.seed(42)
    n = 300
    statuses = ["Approved", "Approved - Conditions", "Cancelled", "Refused"]
    categories = ["Residential", "Commercial", "Industrial"]
    desc_templates = [
        "new single detached house with attached garage",
        "change of use from office to retail store",
        "addition to existing commercial building",
        "demolition of existing structure and rebuild",
        "new multi family residential apartment building",
        "interior alterations to restaurant space",
        "exterior renovation of heritage building facade",
        "new mixed use development with retail ground floor",
        "secondary suite development in basement level",
        "sign permit for new business frontage display",
    ]
    df = pd.DataFrame({
        "applieddate": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "statuscurrent": np.random.choice(statuses, n, p=[0.4, 0.3, 0.2, 0.1]),
        "description": np.random.choice(desc_templates, n),
        "category": np.random.choice(categories, n),
        "landusedistrict": np.random.choice(["R-C1", "C-COR1", "M-CG"], n),
        "communityname": np.random.choice(["Beltline", "Downtown"], n),
        "quadrant": np.random.choice(["NW", "NE", "SW", "SE"], n),
        "permitteddiscretionary": np.random.choice(["Permitted", "Discretionary"], n),
        "latitude": np.random.uniform(51.0, 51.1, n),
        "longitude": np.random.uniform(-114.1, -114.0, n),
    })
    return preprocess(df)


def test_feature_builder_fit_transform(preprocessed_df):
    """FeatureBuilder should produce a sparse matrix with correct row count."""
    fb = FeatureBuilder(tfidf_max_features=50)
    X = fb.fit_transform(preprocessed_df)
    assert X.shape[0] == len(preprocessed_df)
    assert X.shape[1] > 0


def test_split_data(preprocessed_df):
    """split_data should produce correct train/test sizes."""
    fb = FeatureBuilder(tfidf_max_features=50)
    X = fb.fit_transform(preprocessed_df)
    y = preprocessed_df["approved"].values.astype(int)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert len(y_train) + len(y_test) == len(y)


def test_evaluate_model_metrics(preprocessed_df):
    """A trained LogisticRegression should produce valid classification metrics."""
    from sklearn.linear_model import LogisticRegression

    fb = FeatureBuilder(tfidf_max_features=50)
    X = fb.fit_transform(preprocessed_df)
    y = preprocessed_df["approved"].values.astype(int)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    metrics = evaluate_model(clf, X_test, y_test)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert np.isfinite(metrics["auc_roc"])
