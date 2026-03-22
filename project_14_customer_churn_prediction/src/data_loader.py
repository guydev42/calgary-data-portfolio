"""
Data loading, cleaning, feature engineering, and train/test splitting
for the telecom customer churn dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_prepare(filepath="data/telco_churn.csv", test_size=0.2, random_state=42):
    """
    Load the telco churn CSV, engineer features, encode categoricals,
    and return train/test splits.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(filepath)

    # --- Handle missing values ---
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"].fillna(df["monthly_charges"] * df["tenure_months"], inplace=True)

    # Drop customer_id (not a feature)
    df.drop(columns=["customer_id"], inplace=True)

    # --- Feature engineering ---

    # Tenure groups
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12", "13-24", "25-48", "49-72"]
    df["tenure_group"] = pd.cut(df["tenure_months"], bins=bins, labels=labels, include_lowest=True)

    # Charges per month of tenure (avoid division by zero)
    df["charges_per_month_tenure"] = np.where(
        df["tenure_months"] > 0,
        df["total_charges"] / df["tenure_months"],
        df["monthly_charges"],
    )

    # Count of support services
    support_cols = ["online_security", "online_backup", "device_protection", "tech_support"]
    df["has_support_services"] = df[support_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    # --- Encode target ---
    df["churn"] = (df["churn"] == "Yes").astype(int)
    y = df["churn"].values

    # --- Encode features ---

    # Binary columns: label encode
    binary_cols = ["gender", "partner", "dependents", "phone_service", "paperless_billing"]
    le_dict = {}
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Multi-class columns: one-hot encode
    multi_cols = [
        "contract", "internet_service", "multiple_lines",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
        "payment_method", "tenure_group",
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Drop target from features
    feature_cols = [c for c in df.columns if c != "churn"]
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
