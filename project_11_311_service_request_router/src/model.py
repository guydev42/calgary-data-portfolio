"""ML model training and evaluation for 311 Service Request Routing."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os


CATEGORICAL_FEATURES = [
    "channel", "ward",
]

NUMERICAL_FEATURES = [
    "hour", "day_of_week", "month", "year",
    "community_request_count", "community_avg_resolution",
    "service_type_frequency",
]

TARGET = "agency_responsible"


def prepare_model_data(df):
    """Prepare feature matrix and target vector for modeling."""
    df = df.copy()

    # Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna("Unknown")
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Encode service_request_type as a feature
    if "service_request_type" in df.columns:
        le = LabelEncoder()
        df["service_request_type_encoded"] = le.fit_transform(
            df["service_request_type"].astype(str)
        )
        label_encoders["service_request_type"] = le

    # Encode target
    target_le = LabelEncoder()
    df[TARGET] = target_le.fit_transform(df[TARGET].astype(str))
    label_encoders["_target"] = target_le

    # Select available features
    all_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + ["service_request_type_encoded"]
    available_features = [c for c in all_features if c in df.columns]

    X = df[available_features].copy()
    y = df[TARGET].copy()

    # Fill missing numerical values with median
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median())

    # Remove any remaining NaN rows
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    return X, y, label_encoders, available_features


def train_models(X, y, random_state=42):
    """Train multiple classification models and return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state, n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, min_samples_split=10, random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
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
            "Weighted F1": f1_score(y_test, y_pred, average="weighted"),
            "Macro F1": f1_score(y_test, y_pred, average="macro"),
        }
        trained_models[name] = model

    return trained_models, results, scaler, X_test, y_test


def get_feature_importance(model, feature_names, model_name="Random Forest"):
    """Extract feature importance from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)
        return importance
    return pd.DataFrame()


def save_model(model, scaler, label_encoders, feature_names, model_dir):
    """Save trained model and preprocessing artifacts."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.joblib"))
    joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))


def load_model(model_dir):
    """Load trained model and preprocessing artifacts."""
    model = joblib.load(os.path.join(model_dir, "best_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    return model, scaler, label_encoders, feature_names
