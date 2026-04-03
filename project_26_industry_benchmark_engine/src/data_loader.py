"""
Load and prepare the industry benchmark dataset for analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "industry_benchmark.csv"

NUMERIC_KPIS = [
    "avg_recognition_frequency",
    "avg_reward_value",
    "budget_per_employee",
    "turnover_rate",
    "engagement_score",
    "eNPS",
    "training_hours_per_employee",
    "promotion_rate",
    "diversity_index",
    "revenue_per_employee",
    "profit_margin",
]

CATEGORY_COLS = ["industry", "company_size", "region"]

# KPIs where lower is better (used for gap analysis direction)
LOWER_IS_BETTER = {"turnover_rate"}


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the benchmark CSV and return a clean DataFrame."""
    p = Path(path) if path else DATA_PATH
    df = pd.read_csv(p)

    # Ensure correct types
    for col in NUMERIC_KPIS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORY_COLS:
        df[col] = df[col].astype("category")

    return df


def get_industries(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique industries."""
    return sorted(df["industry"].unique())


def get_kpi_display_names() -> dict[str, str]:
    """Map column names to human-readable labels."""
    return {
        "avg_recognition_frequency": "Recognition frequency (events/emp/month)",
        "avg_reward_value": "Avg reward value ($)",
        "budget_per_employee": "Budget per employee ($)",
        "turnover_rate": "Turnover rate",
        "engagement_score": "Engagement score (1-10)",
        "eNPS": "Employee NPS",
        "training_hours_per_employee": "Training hours per employee",
        "promotion_rate": "Promotion rate",
        "diversity_index": "Diversity index",
        "revenue_per_employee": "Revenue per employee ($)",
        "profit_margin": "Profit margin",
    }


def filter_peers(
    df: pd.DataFrame,
    industry: str | None = None,
    company_size: str | None = None,
    region: str | None = None,
) -> pd.DataFrame:
    """Filter dataset to a peer group based on optional criteria."""
    mask = pd.Series(True, index=df.index)
    if industry:
        mask &= df["industry"] == industry
    if company_size:
        mask &= df["company_size"] == company_size
    if region:
        mask &= df["region"] == region
    return df[mask].copy()
