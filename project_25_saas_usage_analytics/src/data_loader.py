"""
Data loading, cleaning, feature engineering, and train/test splitting
for the SaaS usage analytics dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_prepare(filepath="data/saas_usage.csv", test_size=0.2, random_state=42):
    """
    Load the SaaS usage CSV, engineer features, encode categoricals,
    and return train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(filepath)

    # Parse dates
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["last_active_date"] = pd.to_datetime(df["last_active_date"])

    # Drop identifiers (not features)
    df.drop(columns=["user_id", "company_id", "last_active_date"], inplace=True)

    # --- Feature engineering ---

    # Account age in days from signup to observation date
    observation_date = pd.Timestamp("2025-10-01")
    df["account_age_days"] = (observation_date - df["signup_date"]).dt.days
    df.drop(columns=["signup_date"], inplace=True)

    # Feature adoption rate (features_used / 25 available)
    df["feature_adoption_rate"] = df["features_used"] / 25.0

    # Engagement score (composite)
    df["engagement_score"] = (
        df["daily_logins"] / 20.0 * 0.3 +
        df["feature_adoption_rate"] * 0.3 +
        df["session_duration_min"] / 60.0 * 0.2 +
        df["monthly_active_days"] / 30.0 * 0.2
    ).clip(0, 1).round(4)

    # Activity intensity (actions per minute)
    df["actions_per_minute"] = np.where(
        df["session_duration_min"] > 0,
        df["actions_per_session"] / df["session_duration_min"],
        0,
    ).round(4)

    # Support burden (tickets per month of account age)
    df["support_rate"] = np.where(
        df["account_age_days"] > 0,
        df["support_tickets"] / (df["account_age_days"] / 30.0),
        df["support_tickets"],
    ).round(4)

    # Login frequency group
    bins = [0, 3, 8, 15, 40]
    labels = ["Low", "Medium", "High", "Power"]
    df["login_group"] = pd.cut(df["daily_logins"], bins=bins, labels=labels, include_lowest=True)

    # --- Encode target ---
    y = df["is_churned"].values

    # --- Encode features ---

    # Plan tier: ordinal encode
    plan_map = {"Free": 0, "Pro": 1, "Enterprise": 2}
    df["plan_tier_encoded"] = df["plan_tier"].map(plan_map)

    # Multi-class columns: one-hot encode
    multi_cols = ["plan_tier", "industry", "login_group"]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Drop target from features
    feature_cols = [c for c in df.columns if c != "is_churned"]
    X = df[feature_cols].values.astype(float)
    feature_names = list(feature_cols)

    # --- Train/test split (stratified) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Features:     {X_train.shape[1]}")
    print(f"Churn rate (train): {y_train.mean():.3f}")
    print(f"Churn rate (test):  {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare()
    print(f"\nFeature names:\n{feature_names}")
