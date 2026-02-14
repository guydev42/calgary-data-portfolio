"""ML model training for Community Crime Classification."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
import joblib
import os


CRIME_RISK_BINS = {
    "Low": (0, 33),
    "Medium": (33, 66),
    "High": (66, 100),
}


def create_risk_labels(community_df, column="total_crimes"):
    """Assign risk labels based on total crime percentiles."""
    df = community_df.copy()
    df = df.dropna(subset=[column])
    percentiles = df[column].rank(pct=True) * 100
    conditions = [
        percentiles <= 33,
        (percentiles > 33) & (percentiles <= 66),
        percentiles > 66,
    ]
    labels = ["Low", "Medium", "High"]
    df["risk_level"] = np.select(conditions, labels, default="Medium")
    return df


def prepare_classification_data(community_df):
    """Prepare features and labels for crime risk classification."""
    df = community_df.copy()

    # Select numeric feature columns (exclude identifiers and target)
    exclude_cols = ["community", "code", "risk_level", "total_crimes"]
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    # Drop rows with too many missing features
    df = df.dropna(subset=["risk_level"])
    X = df[feature_cols].fillna(0)
    y = df["risk_level"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le, feature_cols


def train_classifiers(X, y, random_state=42):
    """Train multiple classifiers and return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state, multi_class="multinomial",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=random_state,
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 (Weighted)": f1_score(y_test, y_pred, average="weighted"),
            "F1 (Macro)": f1_score(y_test, y_pred, average="macro"),
        }
        trained_models[name] = model

    return trained_models, results, scaler, X_test, y_test


def get_feature_importance(model, feature_names):
    """Extract feature importance from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        return importance
    return pd.DataFrame()


def save_model(model, scaler, label_encoder, feature_names, model_dir):
    """Save trained model and preprocessing artifacts."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))
