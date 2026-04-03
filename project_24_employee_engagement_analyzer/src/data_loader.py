"""
Data loading, cleaning, feature engineering, and train/test splitting
for the employee engagement dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_prepare(filepath="data/employee_engagement.csv", test_size=0.2, random_state=42):
    """
    Load the employee engagement CSV, engineer features, encode categoricals,
    and return train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(filepath)

    # --- Handle missing values ---
    df["satisfaction_survey"] = pd.to_numeric(df["satisfaction_survey"], errors="coerce")
    df["satisfaction_survey"].fillna(df["satisfaction_survey"].median(), inplace=True)

    # Drop employee_id (not a feature) and engagement_score (leaks into target)
    df.drop(columns=["employee_id", "engagement_score"], inplace=True)

    # --- Feature engineering ---

    # Tenure groups
    bins = [0, 6, 12, 24, 48, 144]
    labels = ["0-6", "7-12", "13-24", "25-48", "49+"]
    df["tenure_group"] = pd.cut(df["tenure_months"], bins=bins, labels=labels, include_lowest=True)

    # Recognition balance (given vs received)
    df["recognition_balance"] = df["recognition_events_given"] - df["recognition_events_received"]

    # Total recognition activity
    df["total_recognition_activity"] = df["recognition_events_given"] + df["recognition_events_received"]

    # Has any recognition flag
    df["has_recognition"] = (df["recognition_events_received"] > 0).astype(int)

    # Absenteeism per tenure month (normalized)
    df["absenteeism_rate"] = np.where(
        df["tenure_months"] > 0,
        df["absenteeism_days"] / (df["tenure_months"] / 12),
        df["absenteeism_days"],
    ).round(2)

    # Reward value per recognition event
    df["reward_per_event"] = np.where(
        df["recognition_events_received"] > 0,
        df["avg_reward_value"] / df["recognition_events_received"],
        0,
    ).round(2)

    # --- Encode target ---
    y = df["is_disengaged"].values

    # --- Encode features ---

    # Multi-class columns: one-hot encode
    multi_cols = ["department", "role_level", "reward_type", "tenure_group"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Drop target from features
    feature_cols = [c for c in df.columns if c != "is_disengaged"]
    X = df[feature_cols].values.astype(float)
    feature_names = list(feature_cols)

    # --- Train/test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Features:     {X_train.shape[1]}")
    print(f"Disengagement rate (train): {y_train.mean():.3f}")
    print(f"Disengagement rate (test):  {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()
    print(f"\nFeature names:\n{feature_names}")
