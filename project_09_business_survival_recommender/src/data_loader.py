"""
Data loader for Calgary Business Survival Analyzer & Location Recommender.

Fetches business licence and civic census data from Calgary Open Data via
the Socrata (sodapy) API, caches results locally as CSV, and engineers
features needed for survival analysis and location recommendation.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sodapy import Socrata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BUSINESS_LICENCES_DATASET = "vdjc-pybd"
CIVIC_CENSUS_DATASET = "vsk6-ghca"
SOCRATA_DOMAIN = "data.calgary.ca"

BUSINESS_LICENCES_CSV = DATA_DIR / "business_licences.csv"
CIVIC_CENSUS_CSV = DATA_DIR / "civic_census.csv"

# Columns we expect from the business-licences endpoint
LICENCE_COLUMNS = [
    "getbusid",
    "tradename",
    "homeoccind",
    "address",
    "comdistcd",
    "comdistnm",
    "licencetypes",
    "first_iss_dt",
    "exp_dt",
    "jobstatusdesc",
    "point",
    "globalid",
]

# Statuses that indicate a business is still "alive"
ACTIVE_STATUSES = {"issued", "renewed", "active"}
INACTIVE_STATUSES = {"cancelled", "expired", "closed"}


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------

def _get_socrata_client() -> Socrata:
    """Return an unauthenticated Socrata client for Calgary Open Data."""
    app_token = os.environ.get("SOCRATA_APP_TOKEN")
    return Socrata(SOCRATA_DOMAIN, app_token, timeout=60)


def fetch_business_licences(limit: int = 50000, force: bool = False) -> pd.DataFrame:
    """Fetch business-licence records from Calgary Open Data.

    Parameters
    ----------
    limit : int
        Maximum number of rows to retrieve from the API.
    force : bool
        If *True*, re-download even when a local cache exists.

    Returns
    -------
    pd.DataFrame
        Raw business-licence data.
    """
    if BUSINESS_LICENCES_CSV.exists() and not force:
        logger.info("Loading cached business licences from %s", BUSINESS_LICENCES_CSV)
        return pd.read_csv(BUSINESS_LICENCES_CSV, low_memory=False)

    logger.info("Fetching business licences from Socrata (%s) ...", BUSINESS_LICENCES_DATASET)
    try:
        client = _get_socrata_client()
        records = client.get(BUSINESS_LICENCES_DATASET, limit=limit)
        client.close()

        df = pd.DataFrame.from_records(records)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(BUSINESS_LICENCES_CSV, index=False)
        logger.info("Fetched and cached %d licence records to %s", len(df), BUSINESS_LICENCES_CSV)
        return df
    except Exception as exc:
        logger.error("Failed to fetch business licences from Socrata API: %s", exc)
        if BUSINESS_LICENCES_CSV.exists():
            logger.warning("Falling back to cached business licence data.")
            return pd.read_csv(BUSINESS_LICENCES_CSV, low_memory=False)
        raise


def fetch_civic_census(limit: int = 50000, force: bool = False) -> pd.DataFrame:
    """Fetch civic-census demographic data from Calgary Open Data.

    Parameters
    ----------
    limit : int
        Maximum number of rows to retrieve from the API.
    force : bool
        If *True*, re-download even when a local cache exists.

    Returns
    -------
    pd.DataFrame
        Raw civic-census data.
    """
    if CIVIC_CENSUS_CSV.exists() and not force:
        logger.info("Loading cached civic census from %s", CIVIC_CENSUS_CSV)
        return pd.read_csv(CIVIC_CENSUS_CSV, low_memory=False)

    logger.info("Fetching civic census from Socrata (%s) ...", CIVIC_CENSUS_DATASET)
    try:
        client = _get_socrata_client()
        records = client.get(CIVIC_CENSUS_DATASET, limit=limit)
        client.close()

        df = pd.DataFrame.from_records(records)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CIVIC_CENSUS_CSV, index=False)
        logger.info("Fetched and cached %d census records to %s", len(df), CIVIC_CENSUS_CSV)
        return df
    except Exception as exc:
        logger.error("Failed to fetch civic census from Socrata API: %s", exc)
        if CIVIC_CENSUS_CSV.exists():
            logger.warning("Falling back to cached civic census data.")
            return pd.read_csv(CIVIC_CENSUS_CSV, low_memory=False)
        raise


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse first_iss_dt and exp_dt to datetime and compute business_age_days."""
    df = df.copy()
    df["first_iss_dt"] = pd.to_datetime(df["first_iss_dt"], errors="coerce")
    df["exp_dt"] = pd.to_datetime(df["exp_dt"], errors="coerce")

    # Business age: use expiry date if available, otherwise today
    # Strip timezone info to avoid tz-aware/tz-naive mismatch, then ensure
    # both sides are proper datetime64 before subtraction.
    today = pd.Timestamp(datetime.today().date())
    first_issued = pd.to_datetime(df["first_iss_dt"], utc=True).dt.tz_localize(None)
    exp_date = pd.to_datetime(df["exp_dt"], utc=True).dt.tz_localize(None)
    reference_date = exp_date.fillna(today)
    df["first_iss_dt"] = first_issued
    df["exp_dt"] = exp_date
    df["business_age_days"] = (reference_date - first_issued).dt.days
    df["business_age_days"] = df["business_age_days"].clip(lower=0)

    # Derived calendar features
    df["issue_year"] = df["first_iss_dt"].dt.year
    df["issue_month"] = df["first_iss_dt"].dt.month

    return df


def _create_survived_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a binary *survived* target from jobstatusdesc."""
    df = df.copy()
    status_lower = df["jobstatusdesc"].astype(str).str.strip().str.lower()
    df["survived"] = np.where(
        status_lower.isin(ACTIVE_STATUSES), 1,
        np.where(status_lower.isin(INACTIVE_STATUSES), 0, np.nan)
    )
    # Drop rows where we could not determine status
    before = len(df)
    df = df.dropna(subset=["survived"])
    df["survived"] = df["survived"].astype(int)
    logger.info("Dropped %d rows with unknown survival status", before - len(df))
    return df


def _extract_business_category(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a simplified business_category from licencetypes."""
    df = df.copy()
    df["business_category"] = (
        df["licencetypes"]
        .astype(str)
        .str.strip()
        .str.split(r"\s*-\s*")
        .str[0]
        .str.title()
    )
    return df


def _encode_home_occupation(df: pd.DataFrame) -> pd.DataFrame:
    """Convert homeoccind to a numeric feature."""
    df = df.copy()
    df["is_home_occupation"] = (
        df["homeoccind"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({"Y": 1, "N": 0})
        .fillna(0)
        .astype(int)
    )
    return df


def _build_community_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate community-level features and merge back.

    Creates:
    - business_count: total businesses in the community
    - business_diversity: number of distinct business categories
    - avg_business_age: mean business age (days) in the community
    """
    df = df.copy()
    community_col = "comdistnm"

    agg = (
        df.groupby(community_col)
        .agg(
            business_count=("getbusid", "count"),
            business_diversity=("business_category", "nunique"),
            avg_business_age=("business_age_days", "mean"),
        )
        .reset_index()
    )
    agg["avg_business_age"] = agg["avg_business_age"].round(1)

    # Drop old community-level columns if they exist from a previous merge
    for col in ["business_count", "business_diversity", "avg_business_age"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df.merge(agg, on=community_col, how="left")
    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_and_preprocess(
    licence_limit: int = 50000,
    census_limit: int = 50000,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end load and feature-engineering pipeline.

    Parameters
    ----------
    licence_limit : int
        Max records to fetch for business licences.
    census_limit : int
        Max records to fetch for civic census.
    force_download : bool
        Re-download data even if a cache exists.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (processed_licences, civic_census)
    """
    # -- Business licences --
    licences = fetch_business_licences(limit=licence_limit, force=force_download)
    licences = _parse_dates(licences)
    licences = _create_survived_column(licences)
    licences = _extract_business_category(licences)
    licences = _encode_home_occupation(licences)
    licences = _build_community_features(licences)

    logger.info(
        "Processed %d licence records with %d features",
        len(licences),
        licences.shape[1],
    )

    # -- Civic census --
    census = fetch_civic_census(limit=census_limit, force=force_download)

    return licences, census


def get_community_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a community-level summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed business-licence data (must contain the columns added by
        :func:`load_and_preprocess`).

    Returns
    -------
    pd.DataFrame
        One row per community with aggregated statistics.
    """
    community_col = "comdistnm"

    summary = (
        df.groupby(community_col)
        .agg(
            total_businesses=("getbusid", "count"),
            active_businesses=("survived", "sum"),
            business_diversity=("business_category", "nunique"),
            avg_business_age_days=("business_age_days", "mean"),
            home_occupation_pct=("is_home_occupation", "mean"),
        )
        .reset_index()
    )
    summary["survival_rate"] = (
        summary["active_businesses"] / summary["total_businesses"]
    ).round(4)
    summary["avg_business_age_days"] = summary["avg_business_age_days"].round(1)
    summary["home_occupation_pct"] = (summary["home_occupation_pct"] * 100).round(1)

    return summary.sort_values("total_businesses", ascending=False).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    licences_df, census_df = load_and_preprocess()
    print(f"Licences shape: {licences_df.shape}")
    print(licences_df.head())
    print(f"\nCensus shape: {census_df.shape}")
    print(census_df.head())
