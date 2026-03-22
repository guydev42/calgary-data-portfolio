"""Data loading and preprocessing for Calgary 311 Service Requests dataset."""

import os
import logging
import pandas as pd
import numpy as np
from sodapy import Socrata

logger = logging.getLogger(__name__)

DATASET_ID = "iahh-g8bj"
DOMAIN = "data.calgary.ca"

COLUMNS_TO_KEEP = [
    "created_date", "closed_date", "service_request_type",
    "status", "agency_responsible", "ward", "community",
    "channel",
]

# Mapping from API column names to expected column names.
# The Socrata API for this dataset uses different field names than our pipeline expects.
API_COLUMN_RENAMES = {
    "requested_date": "created_date",
    "service_name": "service_request_type",
    "status_description": "status",
    "source": "channel",
    "comm_name": "community",
    "comm_code": "ward",
}


def fetch_311_requests(limit=100000):
    """Fetch 311 service request data from Calgary Open Data API."""
    logger.info("Fetching 311 data from Socrata API (dataset %s)...", DATASET_ID)
    try:
        client = Socrata(DOMAIN, None, timeout=60)
        results = client.get(DATASET_ID, limit=limit)
        client.close()
        logger.info("Fetched %d records from API.", len(results))
        df = pd.DataFrame.from_records(results)
        # Rename API columns to the names expected by our pipeline
        df = df.rename(columns=API_COLUMN_RENAMES)
        return df
    except Exception as exc:
        logger.error("Failed to fetch 311 data from Socrata API: %s", exc)
        raise


def _normalize_columns(df):
    """Apply column renames so both cached CSVs and fresh API data use consistent names."""
    return df.rename(columns=API_COLUMN_RENAMES)


def load_or_fetch_data(data_dir, limit=100000, force_refresh=False):
    """Load data from local CSV or fetch from API if not available."""
    csv_path = os.path.join(data_dir, "311_requests.csv")
    if os.path.exists(csv_path) and not force_refresh:
        logger.info("Loading cached 311 data from %s", csv_path)
        df = pd.read_csv(csv_path, low_memory=False)
        df = _normalize_columns(df)
        logger.info("Loaded %d records from cache.", len(df))
        return df

    try:
        df = fetch_311_requests(limit=limit)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Cached %d records to %s", len(df), csv_path)
    except Exception as exc:
        logger.error("API fetch failed: %s", exc)
        if os.path.exists(csv_path):
            logger.warning("Falling back to cached 311 data.")
            df = pd.read_csv(csv_path, low_memory=False)
            return _normalize_columns(df)
        raise
    return df


def preprocess_data(df):
    """Clean and preprocess 311 service requests data for modeling."""
    df = df.copy()

    # Keep relevant columns that exist
    available_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df[available_cols]

    # Parse dates
    for date_col in ["created_date", "closed_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Compute resolution time in hours
    if "created_date" in df.columns and "closed_date" in df.columns:
        df["resolution_hours"] = (
            (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
        )
        # Remove negative or extremely large resolution times
        df.loc[df["resolution_hours"] < 0, "resolution_hours"] = np.nan
        df.loc[df["resolution_hours"] > 8760, "resolution_hours"] = np.nan  # > 1 year

    # Extract temporal features from created_date
    if "created_date" in df.columns:
        df["year"] = df["created_date"].dt.year
        df["month"] = df["created_date"].dt.month
        df["day_of_week"] = df["created_date"].dt.dayofweek
        df["hour"] = df["created_date"].dt.hour

    # Drop rows missing key classification fields
    df = df.dropna(subset=["agency_responsible", "service_request_type"])

    # Filter to top 15 departments to avoid rare classes
    top_departments = df["agency_responsible"].value_counts().head(15).index
    df = df[df["agency_responsible"].isin(top_departments)]

    # Fill missing categorical values
    for col in ["channel", "ward", "community"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


def engineer_features(df):
    """Create features for ML modeling."""
    df = df.copy()

    # Community-level aggregate features
    if "community" in df.columns:
        community_stats = df.groupby("community").agg(
            community_request_count=("community", "count"),
        ).reset_index()

        if "resolution_hours" in df.columns:
            resolution_stats = df.groupby("community")["resolution_hours"].mean()
            resolution_stats = resolution_stats.reset_index()
            resolution_stats.columns = ["community", "community_avg_resolution"]
            community_stats = community_stats.merge(
                resolution_stats, on="community", how="left"
            )
        else:
            community_stats["community_avg_resolution"] = 0.0

        df = df.merge(community_stats, on="community", how="left")

    # Service type frequency
    if "service_request_type" in df.columns:
        type_freq = df["service_request_type"].value_counts()
        df["service_type_frequency"] = df["service_request_type"].map(type_freq)

    # Channel encoding (already handled as categorical in model)

    return df
