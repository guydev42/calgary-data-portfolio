"""Data loading and preprocessing for Calgary Building Permits dataset."""

import os
import pandas as pd
import numpy as np
from sodapy import Socrata


DATASET_ID = "c2es-76ed"
DOMAIN = "data.calgary.ca"

COLUMNS_TO_KEEP = [
    "permitnum", "permittype", "permitclass", "permitclassgroup",
    "workclass", "workclassgroup", "statuscurrent",
    "applieddate", "issueddate", "completeddate",
    "description", "housingunits", "estprojectcost",
    "totalsqft", "communitycode", "communityname",
    "latitude", "longitude",
]


def fetch_building_permits(limit=100000):
    """Fetch building permit data from Calgary Open Data API."""
    client = Socrata(DOMAIN, None)
    results = client.get(DATASET_ID, limit=limit)
    client.close()
    df = pd.DataFrame.from_records(results)
    return df


def load_or_fetch_data(data_dir, limit=100000):
    """Load data from local CSV or fetch from API if not available."""
    csv_path = os.path.join(data_dir, "building_permits.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        df = fetch_building_permits(limit=limit)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def preprocess_data(df):
    """Clean and preprocess building permits data for modeling."""
    df = df.copy()

    # Keep relevant columns that exist
    available_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df[available_cols]

    # Convert cost to numeric
    if "estprojectcost" in df.columns:
        df["estprojectcost"] = pd.to_numeric(df["estprojectcost"], errors="coerce")

    # Convert square footage to numeric
    if "totalsqft" in df.columns:
        df["totalsqft"] = pd.to_numeric(df["totalsqft"], errors="coerce")

    # Convert housing units to numeric
    if "housingunits" in df.columns:
        df["housingunits"] = pd.to_numeric(df["housingunits"], errors="coerce")
        df["housingunits"] = df["housingunits"].fillna(0)

    # Parse dates
    for date_col in ["applieddate", "issueddate", "completeddate"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Remove rows without cost or with zero/negative cost
    df = df.dropna(subset=["estprojectcost"])
    df = df[df["estprojectcost"] > 0]

    # Remove extreme outliers (top/bottom 1%)
    lower = df["estprojectcost"].quantile(0.01)
    upper = df["estprojectcost"].quantile(0.99)
    df = df[(df["estprojectcost"] >= lower) & (df["estprojectcost"] <= upper)]

    # Extract date features from applieddate
    if "applieddate" in df.columns:
        df["apply_year"] = df["applieddate"].dt.year
        df["apply_month"] = df["applieddate"].dt.month
        df["apply_dayofweek"] = df["applieddate"].dt.dayofweek

    # Convert latitude/longitude to numeric
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def engineer_features(df):
    """Create features for ML modeling."""
    df = df.copy()

    # Log transform of target variable
    df["log_cost"] = np.log1p(df["estprojectcost"])

    # Log transform of square footage
    if "totalsqft" in df.columns:
        df["log_sqft"] = np.log1p(df["totalsqft"].fillna(0))

    # Cost per square foot (for analysis, not as a feature for prediction)
    if "totalsqft" in df.columns:
        mask = df["totalsqft"] > 0
        df.loc[mask, "cost_per_sqft"] = (
            df.loc[mask, "estprojectcost"] / df.loc[mask, "totalsqft"]
        )

    # Community-level aggregate features
    if "communityname" in df.columns:
        community_stats = df.groupby("communityname")["estprojectcost"].agg(
            ["mean", "median", "count"]
        )
        community_stats.columns = [
            "community_avg_cost",
            "community_median_cost",
            "community_permit_count",
        ]
        df = df.merge(community_stats, left_on="communityname", right_index=True, how="left")

    return df
