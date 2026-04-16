"""
Model training and evaluation for Wildfire Risk Forecaster.
"""

import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    precision_recall_curve, confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from prophet import Prophet


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_config():
    with open(PROJECT_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def train_risk_classifier(X, y):
    """Train XGBoost and Random Forest classifiers, return best."""
    config = load_config()
    test_size = config["model"]["test_size"]
    seed = config["model"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    models = {
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="logloss",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=seed,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "auc_roc": auc,
            "f1_score": f1,
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    return results, best_name


def forecast_risk_timeseries(df, periods=90):
    """
    Use Prophet to forecast fire risk index over time.
    Expects df with columns: ds (date), y (risk_index).
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
    )
    model.fit(df[["ds", "y"]])

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast, model


def load_trained_model():
    """Load the saved best model."""
    path = MODELS_DIR / "best_model.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def predict_risk(model, X):
    """Predict fire risk probability for new data."""
    proba = model.predict_proba(X)[:, 1]
    labels = (proba >= 0.5).astype(int)
    return labels, proba
