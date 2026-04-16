"""
Anomaly detection models for Environmental Anomaly Detector.
"""

import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_config():
    with open(PROJECT_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def train_isolation_forest(X):
    """Train Isolation Forest for anomaly detection."""
    config = load_config()
    contamination = config["model"]["contamination"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=config["model"]["random_state"],
        n_jobs=-1,
    )
    model.fit(X_scaled)

    joblib.dump(model, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    scores = model.decision_function(X_scaled)
    labels = model.predict(X_scaled)
    return labels, scores


def train_lof(X):
    """Train Local Outlier Factor."""
    config = load_config()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=config["model"]["contamination"],
        novelty=False,
    )
    labels = lof.fit_predict(X_scaled)
    scores = lof.negative_outlier_factor_
    return labels, scores


def ensemble_vote(if_labels, lof_labels, ae_labels=None):
    """Majority vote across detection methods."""
    votes = np.column_stack([if_labels, lof_labels])
    if ae_labels is not None:
        votes = np.column_stack([votes, ae_labels])

    anomaly_votes = (votes == -1).sum(axis=1)
    threshold = votes.shape[1] / 2
    ensemble_labels = np.where(anomaly_votes > threshold, -1, 1)
    return ensemble_labels


def load_trained_model(name="isolation_forest"):
    """Load a saved model."""
    path = MODELS_DIR / f"{name}.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def detect_anomalies(model, scaler, X):
    """Detect anomalies on new data."""
    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)
    return labels, scores
