"""
Core benchmarking engine: percentile computation, peer comparison, and gap analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats

from .data_loader import NUMERIC_KPIS, LOWER_IS_BETTER, get_kpi_display_names


# ── Percentile computation ───────────────────────────────────────────

def compute_percentiles(df: pd.DataFrame, kpi: str) -> pd.Series:
    """Return the percentile rank (0-100) for each row on a given KPI."""
    return df[kpi].rank(pct=True).mul(100).round(1)


def compute_all_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentile columns for every numeric KPI."""
    result = df.copy()
    for kpi in NUMERIC_KPIS:
        result[f"{kpi}_pctl"] = compute_percentiles(df, kpi)
    return result


def company_percentile_card(
    df: pd.DataFrame, company_id: str
) -> pd.DataFrame:
    """
    For a single company, return a DataFrame with columns:
    KPI | Value | Percentile | Industry median | vs Median
    """
    pctl_df = compute_all_percentiles(df)
    row = pctl_df[pctl_df["company_id"] == company_id]
    if row.empty:
        raise ValueError(f"Company {company_id} not found.")
    row = row.iloc[0]
    industry = row["industry"]
    ind_medians = df[df["industry"] == industry][NUMERIC_KPIS].median()

    display = get_kpi_display_names()
    records = []
    for kpi in NUMERIC_KPIS:
        val = row[kpi]
        pctl = row[f"{kpi}_pctl"]
        med = ind_medians[kpi]
        diff = val - med
        records.append({
            "KPI": display.get(kpi, kpi),
            "Value": val,
            "Percentile": pctl,
            "Industry median": round(med, 3),
            "vs Median": round(diff, 3),
        })
    return pd.DataFrame(records)


# ── Industry summary ─────────────────────────────────────────────────

def industry_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean, median, P25, P75 for each KPI by industry."""
    records = []
    for industry, grp in df.groupby("industry"):
        for kpi in NUMERIC_KPIS:
            vals = grp[kpi].dropna()
            records.append({
                "industry": industry,
                "kpi": kpi,
                "mean": vals.mean(),
                "median": vals.median(),
                "p25": vals.quantile(0.25),
                "p75": vals.quantile(0.75),
                "std": vals.std(),
                "count": len(vals),
            })
    return pd.DataFrame(records).round(3)


# ── Peer comparison ──────────────────────────────────────────────────

def peer_comparison(
    df: pd.DataFrame,
    company_id: str,
    peer_filter: dict | None = None,
) -> pd.DataFrame:
    """
    Compare a company against its peers.
    peer_filter: optional dict with keys 'industry', 'company_size', 'region'.
    Returns a DataFrame with the company value, peer median, percentile, and gap.
    """
    row = df[df["company_id"] == company_id]
    if row.empty:
        raise ValueError(f"Company {company_id} not found.")
    row = row.iloc[0]

    # Build peer group
    peers = df.copy()
    if peer_filter:
        for col, val in peer_filter.items():
            if val:
                peers = peers[peers[col] == val]
    else:
        peers = peers[peers["industry"] == row["industry"]]

    display = get_kpi_display_names()
    records = []
    for kpi in NUMERIC_KPIS:
        val = row[kpi]
        peer_vals = peers[kpi].dropna()
        pctl = stats.percentileofscore(peer_vals, val, kind="rank")
        peer_med = peer_vals.median()
        gap = val - peer_med
        direction = "above" if gap > 0 else "below" if gap < 0 else "at"
        if kpi in LOWER_IS_BETTER:
            direction = "below" if gap < 0 else "above" if gap > 0 else "at"
            favorable = gap < 0
        else:
            favorable = gap > 0

        records.append({
            "KPI": display.get(kpi, kpi),
            "Your value": round(val, 3),
            "Peer median": round(peer_med, 3),
            "Gap": round(gap, 3),
            "Percentile": round(pctl, 1),
            "Direction": direction,
            "Favorable": favorable,
            "Peer count": len(peer_vals),
        })
    return pd.DataFrame(records)


# ── Gap analysis ─────────────────────────────────────────────────────

def gap_analysis(
    df: pd.DataFrame,
    company_id: str,
    target_percentile: float = 75.0,
) -> pd.DataFrame:
    """
    Identify gaps between a company's current KPIs and the target percentile
    within their industry. Returns actionable improvement targets.
    """
    row = df[df["company_id"] == company_id]
    if row.empty:
        raise ValueError(f"Company {company_id} not found.")
    row = row.iloc[0]
    industry = row["industry"]
    peers = df[df["industry"] == industry]

    display = get_kpi_display_names()
    records = []
    for kpi in NUMERIC_KPIS:
        val = row[kpi]
        target_val = peers[kpi].quantile(target_percentile / 100)
        current_pctl = stats.percentileofscore(peers[kpi].dropna(), val, kind="rank")

        if kpi in LOWER_IS_BETTER:
            # For turnover, target is the lower percentile
            target_val = peers[kpi].quantile(1 - target_percentile / 100)
            needs_improvement = val > target_val
            gap = val - target_val
        else:
            needs_improvement = val < target_val
            gap = target_val - val

        records.append({
            "KPI": display.get(kpi, kpi),
            "Current value": round(val, 3),
            "Current percentile": round(current_pctl, 1),
            "Target (P{:.0f})".format(target_percentile): round(target_val, 3),
            "Gap to target": round(gap, 3),
            "Needs improvement": needs_improvement,
            "Priority": "High" if needs_improvement and abs(gap) > peers[kpi].std() else
                        "Medium" if needs_improvement else "On track",
        })
    return pd.DataFrame(records)


# ── Cross-industry ranking ───────────────────────────────────────────

def cross_industry_ranking(df: pd.DataFrame, kpi: str) -> pd.DataFrame:
    """Rank industries by a given KPI (median)."""
    ranking = (
        df.groupby("industry")[kpi]
        .agg(["median", "mean", "std", "count"])
        .sort_values("median", ascending=(kpi in LOWER_IS_BETTER))
        .reset_index()
    )
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking
