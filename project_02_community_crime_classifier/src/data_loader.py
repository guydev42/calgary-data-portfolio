"""Data loading and preprocessing for Calgary Community Crime Statistics."""

import os
import pandas as pd
import numpy as np
from sodapy import Socrata

DOMAIN = "data.calgary.ca"
CRIME_DATASET_ID = "78gh-n26t"
CENSUS_DATASET_ID = "vsk6-ghca"


def fetch_crime_data(limit=100000):
    """Fetch community crime statistics from Calgary Open Data API."""
    client = Socrata(DOMAIN, None)
    results = client.get(CRIME_DATASET_ID, limit=limit)
    client.close()
    return pd.DataFrame.from_records(results)


def fetch_census_data(limit=100000):
    """Fetch civic census demographics data."""
    client = Socrata(DOMAIN, None)
    results = client.get(CENSUS_DATASET_ID, limit=limit)
    client.close()
    return pd.DataFrame.from_records(results)


def load_or_fetch_crime_data(data_dir, limit=100000):
    """Load crime data from local CSV or fetch from API."""
    csv_path = os.path.join(data_dir, "crime_statistics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        df = fetch_crime_data(limit=limit)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def load_or_fetch_census_data(data_dir, limit=100000):
    """Load census data from local CSV or fetch from API."""
    csv_path = os.path.join(data_dir, "census_demographics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
    else:
        df = fetch_census_data(limit=limit)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def preprocess_crime_data(df):
    """Clean and preprocess crime statistics."""
    df = df.copy()
    df["crime_count"] = pd.to_numeric(df["crime_count"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df = df.dropna(subset=["crime_count", "year", "community", "category"])
    df["crime_count"] = df["crime_count"].astype(int)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df


def preprocess_census_data(df):
    """Clean and preprocess census demographics."""
    df = df.copy()
    for col in ["males", "females", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["code", "year"])
    df["total_pop"] = df["males"].fillna(0) + df["females"].fillna(0)
    return df


def create_community_features(crime_df, census_df):
    """Create community-level features combining crime and demographic data."""
    # Crime features per community
    crime_agg = crime_df.groupby("community").agg(
        total_crimes=("crime_count", "sum"),
        avg_monthly_crimes=("crime_count", "mean"),
        crime_categories=("category", "nunique"),
        years_of_data=("year", "nunique"),
    ).reset_index()

    # Crime category breakdown
    crime_pivot = crime_df.groupby(["community", "category"])["crime_count"].sum().unstack(fill_value=0)
    crime_pivot.columns = [f"crime_{c.lower().replace(' ', '_').replace('&', 'and').replace('-', '_').replace('(', '').replace(')', '')}" for c in crime_pivot.columns]
    crime_pivot = crime_pivot.reset_index()

    # Merge
    community_df = crime_agg.merge(crime_pivot, on="community", how="left")

    # Add census demographics if available
    if census_df is not None and len(census_df) > 0:
        latest_year = census_df["year"].max()
        census_latest = census_df[census_df["year"] == latest_year]
        census_agg = census_latest.groupby("code").agg(
            total_population=("total_pop", "sum"),
            male_population=("males", "sum"),
            female_population=("females", "sum"),
        ).reset_index()
        community_df = community_df.merge(
            census_agg, left_on="community", right_on="code", how="left"
        )
        # Crime rate per capita
        mask = community_df["total_population"] > 0
        community_df.loc[mask, "crime_rate_per_1000"] = (
            community_df.loc[mask, "total_crimes"]
            / community_df.loc[mask, "total_population"]
            * 1000
        )

    return community_df


def create_temporal_crime_data(crime_df):
    """Create time-series crime data for trend analysis."""
    temporal = crime_df.groupby(["year", "month", "category"])["crime_count"].sum().reset_index()
    temporal["date"] = pd.to_datetime(
        temporal["year"].astype(str) + "-" + temporal["month"].astype(str).str.zfill(2) + "-01"
    )
    return temporal
