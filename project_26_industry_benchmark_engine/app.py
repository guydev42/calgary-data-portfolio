"""
Streamlit dashboard for the industry benchmark engine.
Five pages: Industry Overview, Company Benchmarker, Percentile Rankings,
Trend Analysis, Custom Report Generator.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")

from src.data_loader import (
    load_data, NUMERIC_KPIS, get_kpi_display_names,
    get_industries, filter_peers,
)
from src.benchmark import (
    compute_all_percentiles,
    company_percentile_card,
    industry_summary,
    peer_comparison,
    gap_analysis,
    cross_industry_ranking,
)

st.set_page_config(page_title="Industry benchmark engine", layout="wide")

DATA_PATH = "data/industry_benchmark.csv"


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_data
def get_summary(_df):
    return industry_summary(_df)


# --- Sidebar navigation ---
page = st.sidebar.radio(
    "Navigate",
    [
        "Industry Overview",
        "Company Benchmarker",
        "Percentile Rankings",
        "Trend Analysis",
        "Custom Report Generator",
    ],
)

df = get_data()
display_names = get_kpi_display_names()
kpi_options = {v: k for k, v in display_names.items()}

# =====================================================================
# PAGE 1: INDUSTRY OVERVIEW
# =====================================================================
if page == "Industry Overview":
    st.title("Industry benchmark overview")
    st.markdown(
        "Compare recognition program KPIs across 8 industries. "
        "Identify which sectors lead in engagement, retention, and ROI."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies", f"{len(df):,}")
    col2.metric("Industries", f"{df['industry'].nunique()}")
    col3.metric("Avg engagement", f"{df['engagement_score'].mean():.1f}/10")
    col4.metric("Median eNPS", f"{int(df['eNPS'].median())}")

    st.subheader("KPI comparison across industries")
    selected_kpi_label = st.selectbox(
        "Select KPI", list(kpi_options.keys()), index=0
    )
    selected_kpi = kpi_options[selected_kpi_label]

    summary = get_summary(df)
    kpi_summary = summary[summary["kpi"] == selected_kpi].sort_values("median", ascending=False)

    fig = px.bar(
        kpi_summary, x="industry", y="median",
        error_y=kpi_summary["p75"] - kpi_summary["median"],
        error_y_minus=kpi_summary["median"] - kpi_summary["p25"],
        color="median",
        color_continuous_scale="Viridis",
        title=f"{selected_kpi_label} by industry (median with IQR)",
        labels={"median": selected_kpi_label, "industry": "Industry"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap of all KPIs
    st.subheader("Cross-industry heatmap")
    pivot = summary.pivot_table(index="industry", columns="kpi", values="median")
    # Normalize for heatmap
    pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min())
    pivot_norm.columns = [display_names.get(c, c) for c in pivot_norm.columns]

    fig2 = px.imshow(
        pivot_norm.round(2),
        text_auto=".2f",
        color_continuous_scale="YlOrRd",
        title="Normalized industry KPI heatmap (0 = lowest, 1 = highest)",
        aspect="auto",
    )
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Dataset preview")
    st.dataframe(df.head(15), use_container_width=True)


# =====================================================================
# PAGE 2: COMPANY BENCHMARKER
# =====================================================================
elif page == "Company Benchmarker":
    st.title("Company benchmarker")
    st.markdown("Select a company and compare its KPIs against industry peers.")

    col1, col2 = st.columns([1, 2])
    with col1:
        company = st.selectbox("Select company", sorted(df["company_id"].unique()))
        company_row = df[df["company_id"] == company].iloc[0]
        st.markdown(f"**Industry:** {company_row['industry']}")
        st.markdown(f"**Size:** {company_row['company_size']}")
        st.markdown(f"**Region:** {company_row['region']}")
        st.markdown(f"**Employees:** {company_row['employee_count']:,}")

    with col2:
        st.subheader("Peer comparison filters")
        peer_industry = st.selectbox(
            "Industry filter",
            ["Same industry"] + get_industries(df),
            index=0,
        )
        peer_size = st.selectbox(
            "Size filter",
            ["All sizes", "Small", "Medium", "Large", "Enterprise"],
            index=0,
        )
        peer_region = st.selectbox(
            "Region filter",
            ["All regions"] + sorted(df["region"].unique()),
            index=0,
        )

    # Build peer filter
    pf = {}
    if peer_industry == "Same industry":
        pf["industry"] = company_row["industry"]
    elif peer_industry != "All industries":
        pf["industry"] = peer_industry
    if peer_size != "All sizes":
        pf["company_size"] = peer_size
    if peer_region != "All regions":
        pf["region"] = peer_region

    comparison = peer_comparison(df, company, peer_filter=pf if pf else None)

    st.subheader("KPI comparison")
    # Color-coded table
    def highlight_favorable(row):
        if row["Favorable"]:
            return ["background-color: #d4edda"] * len(row)
        elif row["Direction"] == "at":
            return [""] * len(row)
        return ["background-color: #f8d7da"] * len(row)

    styled = comparison.style.apply(highlight_favorable, axis=1).format({
        "Your value": "{:.3f}",
        "Peer median": "{:.3f}",
        "Gap": "{:+.3f}",
        "Percentile": "{:.1f}",
    })
    st.dataframe(styled, use_container_width=True)

    # Radar chart
    st.subheader("Percentile radar")
    radar_kpis = comparison["KPI"].tolist()
    radar_pctls = comparison["Percentile"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=radar_pctls + [radar_pctls[0]],
        theta=radar_kpis + [radar_kpis[0]],
        fill="toself",
        name=company,
        line=dict(color="#636EFA"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=[50] * (len(radar_kpis) + 1),
        theta=radar_kpis + [radar_kpis[0]],
        name="Median (P50)",
        line=dict(color="#EF553B", dash="dash"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=f"Percentile profile: {company}",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE 3: PERCENTILE RANKINGS
# =====================================================================
elif page == "Percentile Rankings":
    st.title("Percentile rankings")
    st.markdown("See how every company ranks within its industry on each KPI.")

    selected_kpi_label = st.selectbox("Select KPI", list(kpi_options.keys()), index=4)
    selected_kpi = kpi_options[selected_kpi_label]

    filter_industry = st.selectbox(
        "Filter by industry", ["All industries"] + get_industries(df)
    )

    view_df = df.copy()
    if filter_industry != "All industries":
        view_df = view_df[view_df["industry"] == filter_industry]

    pctl_df = compute_all_percentiles(view_df)
    pctl_col = f"{selected_kpi}_pctl"

    ranking = pctl_df[["company_id", "industry", "company_size", selected_kpi, pctl_col]].sort_values(
        pctl_col, ascending=False
    ).reset_index(drop=True)
    ranking.index += 1
    ranking.index.name = "Rank"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(ranking.head(50), use_container_width=True)

    with col2:
        # Distribution plot
        fig = px.histogram(
            view_df, x=selected_kpi, color="industry",
            barmode="overlay", nbins=30,
            title=f"{selected_kpi_label} distribution",
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Industry ranking
    st.subheader("Industry ranking")
    ind_ranking = cross_industry_ranking(df, selected_kpi)
    fig2 = px.bar(
        ind_ranking, x="industry", y="median",
        color="median",
        color_continuous_scale="Viridis",
        title=f"Industry ranking by {selected_kpi_label}",
    )
    st.plotly_chart(fig2, use_container_width=True)


# =====================================================================
# PAGE 4: TREND ANALYSIS
# =====================================================================
elif page == "Trend Analysis":
    st.title("Trend analysis")
    st.markdown(
        "Explore relationships between KPIs. "
        "Identify which recognition program metrics drive business outcomes."
    )

    col1, col2 = st.columns(2)
    with col1:
        x_label = st.selectbox("X-axis KPI", list(kpi_options.keys()), index=0)
        x_kpi = kpi_options[x_label]
    with col2:
        y_label = st.selectbox("Y-axis KPI", list(kpi_options.keys()), index=3)
        y_kpi = kpi_options[y_label]

    color_by = st.selectbox("Color by", ["industry", "company_size", "region"])

    fig = px.scatter(
        df, x=x_kpi, y=y_kpi,
        color=color_by,
        size="employee_count",
        hover_data=["company_id", "industry", "company_size"],
        title=f"{x_label} vs {y_label}",
        trendline="ols",
        opacity=0.7,
    )
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.subheader("KPI correlation matrix")
    corr = df[NUMERIC_KPIS].corr()
    corr.columns = [display_names.get(c, c) for c in corr.columns]
    corr.index = [display_names.get(c, c) for c in corr.index]

    fig2 = px.imshow(
        corr.round(2), text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Pairwise KPI correlations",
        aspect="auto",
    )
    fig2.update_layout(height=550)
    st.plotly_chart(fig2, use_container_width=True)

    # Box plot comparison
    st.subheader("Distribution by industry")
    box_kpi_label = st.selectbox("KPI for box plot", list(kpi_options.keys()), index=4, key="box")
    box_kpi = kpi_options[box_kpi_label]

    fig3 = px.box(
        df, x="industry", y=box_kpi, color="industry",
        title=f"{box_kpi_label} distribution by industry",
    )
    fig3.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)


# =====================================================================
# PAGE 5: CUSTOM REPORT GENERATOR
# =====================================================================
elif page == "Custom Report Generator":
    st.title("Custom report generator")
    st.markdown("Generate a benchmarking report for any company with gap analysis and improvement targets.")

    company = st.selectbox("Select company", sorted(df["company_id"].unique()), key="report_company")
    target_pctl = st.slider("Target percentile", 50, 95, 75, step=5)

    company_row = df[df["company_id"] == company].iloc[0]
    st.markdown(f"**{company}** | {company_row['industry']} | {company_row['company_size']} | {company_row['region']} | {company_row['employee_count']:,} employees")

    gaps = gap_analysis(df, company, target_percentile=target_pctl)

    # Summary metrics
    high_priority = (gaps["Priority"] == "High").sum()
    medium_priority = (gaps["Priority"] == "Medium").sum()
    on_track = (gaps["Priority"] == "On track").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("High priority gaps", high_priority, delta_color="inverse")
    col2.metric("Medium priority gaps", medium_priority)
    col3.metric("On track", on_track)

    st.subheader("Gap analysis")

    def highlight_priority(row):
        if row["Priority"] == "High":
            return ["background-color: #f8d7da"] * len(row)
        elif row["Priority"] == "Medium":
            return ["background-color: #fff3cd"] * len(row)
        return ["background-color: #d4edda"] * len(row)

    styled = gaps.style.apply(highlight_priority, axis=1)
    st.dataframe(styled, use_container_width=True)

    # Improvement roadmap
    st.subheader("Improvement roadmap")
    high_gaps = gaps[gaps["Priority"] == "High"].sort_values("Gap to target", ascending=False)
    if not high_gaps.empty:
        for _, row in high_gaps.iterrows():
            target_col = [c for c in row.index if c.startswith("Target")][0]
            st.markdown(
                f"- **{row['KPI']}**: currently at {row['Current value']:.3f} "
                f"(P{row['Current percentile']:.0f}), target {row[target_col]:.3f} "
                f"(P{target_pctl}). Gap: **{row['Gap to target']:.3f}**"
            )
    else:
        st.success("All KPIs meet or exceed the target percentile.")

    # Percentile card
    st.subheader("Full percentile card")
    card = company_percentile_card(df, company)
    st.dataframe(
        card.style.format({
            "Value": "{:.3f}",
            "Percentile": "{:.1f}",
            "Industry median": "{:.3f}",
            "vs Median": "{:+.3f}",
        }),
        use_container_width=True,
    )

    # Export
    st.subheader("Export report")
    csv_data = gaps.to_csv(index=False)
    st.download_button(
        label="Download gap analysis as CSV",
        data=csv_data,
        file_name=f"benchmark_report_{company}.csv",
        mime="text/csv",
    )
