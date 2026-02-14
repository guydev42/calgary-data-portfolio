"""
Calgary Business Survival Analyzer & Location Recommender -- Streamlit App.

Provides an interactive dashboard for exploring Calgary business-licence data,
running survival analysis, and generating community-level location
recommendations for prospective business owners.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on the path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_preprocess, get_community_summary  # noqa: E402
from src.model import (  # noqa: E402
    SurvivalAnalyzer,
    recommend_locations,
    get_competition_analysis,
    train_random_forest,
    train_xgboost,
    save_model,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Calgary Business Survival Analyzer & Location Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Business Dashboard",
        "Survival Analysis",
        "Location Recommender",
        "Model Performance",
        "About",
    ],
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading Calgary business data ...")
def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess licence + census data."""
    licences, census = load_and_preprocess()
    return licences, census


@st.cache_data(show_spinner="Building community summary ...")
def _community_summary(df: pd.DataFrame) -> pd.DataFrame:
    return get_community_summary(df)


@st.cache_resource(show_spinner="Training Random Forest ...")
def _train_rf(df: pd.DataFrame) -> dict:
    return train_random_forest(df)


@st.cache_resource(show_spinner="Training XGBoost ...")
def _train_xgb(df: pd.DataFrame) -> dict:
    return train_xgboost(df)


# ---------------------------------------------------------------------------
# Helper: generate synthetic demo data if API unavailable
# ---------------------------------------------------------------------------

def _generate_demo_data() -> pd.DataFrame:
    """Create a realistic synthetic dataset for demonstration purposes."""
    rng = np.random.RandomState(42)
    n = 5000

    communities = [
        "BELTLINE", "DOWNTOWN COMMERCIAL CORE", "KENSINGTON", "INGLEWOOD",
        "BRIDGELAND/RIVERSIDE", "MISSION", "SUNALTA", "HILLHURST",
        "MOUNT ROYAL", "ALTADORE", "MARDA LOOP", "BOWNESS",
        "FOREST LAWN", "MARLBOROUGH", "SADDLE RIDGE", "PANORAMA HILLS",
        "TUSCANY", "CRANSTON", "AUBURN BAY", "SILVERADO",
    ]

    categories = [
        "Food Services", "Retail", "Professional Services", "Health Services",
        "Construction", "Transportation", "Education", "Entertainment",
        "Technology", "Personal Services", "Financial Services", "Wholesale",
    ]

    statuses = ["active", "renewed", "cancelled", "expired"]
    status_weights = [0.45, 0.20, 0.20, 0.15]

    start_dates = pd.date_range("2010-01-01", "2024-12-31", periods=n)
    ages = rng.exponential(scale=1200, size=n).clip(30, 5500).astype(int)
    issue_years = pd.Series(start_dates).dt.year.values
    issue_months = pd.Series(start_dates).dt.month.values
    chosen_statuses = rng.choice(statuses, size=n, p=status_weights)

    df = pd.DataFrame({
        "getbusid": [f"BUS{i:06d}" for i in range(n)],
        "tradename": [f"Business {i}" for i in range(n)],
        "comdistnm": rng.choice(communities, size=n),
        "business_category": rng.choice(categories, size=n),
        "jobstatusdesc": chosen_statuses,
        "first_iss_dt": start_dates,
        "exp_dt": start_dates + pd.to_timedelta(ages, unit="D"),
        "business_age_days": ages,
        "issue_year": issue_years,
        "issue_month": issue_months,
        "is_home_occupation": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
        "homeoccind": rng.choice(["Y", "N"], size=n, p=[0.3, 0.7]),
    })

    active_statuses = {"active", "renewed"}
    df["survived"] = df["jobstatusdesc"].isin(active_statuses).astype(int)

    # Event observed: 1 if closed/cancelled (for survival analysis the "event"
    # is business closure), 0 if still alive (censored)
    df["event_observed"] = 1 - df["survived"]

    # Community-level features
    comm_agg = (
        df.groupby("comdistnm")
        .agg(
            business_count=("getbusid", "count"),
            business_diversity=("business_category", "nunique"),
            avg_business_age=("business_age_days", "mean"),
        )
        .reset_index()
    )
    comm_agg["avg_business_age"] = comm_agg["avg_business_age"].round(1)
    df = df.merge(comm_agg, on="comdistnm", how="left")

    return df


# ---------------------------------------------------------------------------
# Data initialisation
# ---------------------------------------------------------------------------

try:
    licences_df, census_df = _load_data()
    # Create event_observed column for survival analysis
    if "event_observed" not in licences_df.columns:
        licences_df["event_observed"] = 1 - licences_df["survived"]
    data_source = "Calgary Open Data API"
except Exception:
    licences_df = _generate_demo_data()
    census_df = pd.DataFrame()
    data_source = "Synthetic demo data (API unavailable)"

st.sidebar.caption(f"Data source: {data_source}")
st.sidebar.caption(f"Records: {len(licences_df):,}")


# ===================================================================
# PAGE: Business Dashboard
# ===================================================================

def page_business_dashboard() -> None:
    """Render the Business Dashboard page."""
    st.title("Calgary Business Dashboard")
    st.markdown("Overview of business-licence activity across the city.")

    # --- KPI row ---------------------------------------------------------
    total = len(licences_df)
    active = int(licences_df["survived"].sum())
    active_rate = active / total if total else 0
    avg_age_years = licences_df["business_age_days"].mean() / 365

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Businesses", f"{total:,}")
    c2.metric("Active Businesses", f"{active:,}")
    c3.metric("Active Rate", f"{active_rate:.1%}")
    c4.metric("Avg Age (years)", f"{avg_age_years:.1f}")

    st.divider()

    # --- Businesses by type ----------------------------------------------
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Businesses by Type")
        type_counts = (
            licences_df["business_category"]
            .value_counts()
            .head(15)
            .reset_index()
        )
        type_counts.columns = ["Business Type", "Count"]
        fig_type = px.bar(
            type_counts,
            x="Count",
            y="Business Type",
            orientation="h",
            color="Count",
            color_continuous_scale="Viridis",
        )
        fig_type.update_layout(
            yaxis=dict(autorange="reversed"),
            height=450,
            showlegend=False,
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with col_right:
        st.subheader("Status Distribution")
        status_counts = (
            licences_df["jobstatusdesc"]
            .str.title()
            .value_counts()
            .reset_index()
        )
        status_counts.columns = ["Status", "Count"]
        fig_pie = px.pie(
            status_counts,
            names="Status",
            values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=450)
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Businesses by community -----------------------------------------
    st.subheader("Top Communities by Number of Businesses")
    comm_counts = (
        licences_df["comdistnm"]
        .value_counts()
        .head(20)
        .reset_index()
    )
    comm_counts.columns = ["Community", "Count"]
    fig_comm = px.bar(
        comm_counts,
        x="Community",
        y="Count",
        color="Count",
        color_continuous_scale="Blues",
    )
    fig_comm.update_layout(xaxis_tickangle=-45, height=450, showlegend=False)
    st.plotly_chart(fig_comm, use_container_width=True)


# ===================================================================
# PAGE: Survival Analysis
# ===================================================================

def page_survival_analysis() -> None:
    """Render the Survival Analysis page."""
    st.title("Business Survival Analysis")
    st.markdown(
        "Kaplan-Meier survival curves and Cox Proportional-Hazards analysis "
        "reveal how long businesses typically survive and which factors "
        "influence their longevity."
    )

    analyzer = SurvivalAnalyzer()

    # --- Kaplan-Meier by business type -----------------------------------
    st.subheader("Kaplan-Meier Survival Curves by Business Type")
    top_n = st.slider("Number of business types to display", 3, 15, 8)

    km_data = analyzer.kaplan_meier_by_group(
        licences_df,
        duration_col="business_age_days",
        event_col="event_observed",
        group_col="business_category",
        top_n=top_n,
    )

    if not km_data.empty:
        # Convert timeline from days to years for readability
        km_data["timeline_years"] = km_data["timeline"] / 365.0
        fig_km = px.line(
            km_data,
            x="timeline_years",
            y="survival_probability",
            color="label",
            labels={
                "timeline_years": "Years Since Licence Issued",
                "survival_probability": "Survival Probability",
                "label": "Business Type",
            },
        )
        fig_km.update_layout(height=500)
        st.plotly_chart(fig_km, use_container_width=True)
    else:
        st.info("Not enough data to generate survival curves.")

    st.divider()

    # --- Survival rates at milestones ------------------------------------
    st.subheader("Survival Rates at 1, 3, and 5 Years")
    milestone_df = analyzer.survival_rates_at_milestones(
        licences_df,
        duration_col="business_age_days",
        event_col="event_observed",
        group_col="business_category",
        top_n=12,
    )
    if not milestone_df.empty:
        st.dataframe(
            milestone_df.style.format({
                "1_year_survival": "{:.1%}",
                "3_year_survival": "{:.1%}",
                "5_year_survival": "{:.1%}",
            }),
            use_container_width=True,
        )
    else:
        st.info("Insufficient data for milestone table.")

    st.divider()

    # --- Cox PH hazard factors -------------------------------------------
    st.subheader("Hazard Factors (Cox Proportional-Hazards Model)")
    cox_features = [
        "business_age_days", "event_observed", "is_home_occupation",
        "business_count", "business_diversity",
    ]
    cox_available = [c for c in cox_features if c in licences_df.columns]

    if len(cox_available) >= 3:
        cox_df = licences_df[cox_available].dropna()
        try:
            analyzer.fit_cox(cox_df, duration_col="business_age_days",
                             event_col="event_observed")
            hazard_summary = analyzer.get_hazard_summary().reset_index()
            hazard_summary = hazard_summary.rename(columns={"index": "covariate"})

            st.dataframe(hazard_summary, use_container_width=True)

            # Visualize hazard ratios
            if "exp(coef)" in hazard_summary.columns:
                fig_hr = px.bar(
                    hazard_summary,
                    x="covariate",
                    y="exp(coef)",
                    color="exp(coef)",
                    color_continuous_scale="RdYlGn_r",
                    labels={"exp(coef)": "Hazard Ratio"},
                    title="Hazard Ratios (>1 = higher risk of closure)",
                )
                fig_hr.add_hline(y=1.0, line_dash="dash", line_color="grey")
                fig_hr.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_hr, use_container_width=True)
        except Exception as exc:
            st.warning(f"Cox model could not be fitted: {exc}")
    else:
        st.info("Not enough features available for Cox PH analysis.")


# ===================================================================
# PAGE: Location Recommender
# ===================================================================

def page_location_recommender() -> None:
    """Render the Location Recommender page."""
    st.title("Business Location Recommender")
    st.markdown(
        "Select a business type to see which Calgary communities offer the "
        "best environment for success based on survival rates, competition, "
        "and diversity."
    )

    # Unique categories sorted alphabetically
    categories = sorted(licences_df["business_category"].dropna().unique().tolist())
    selected_type = st.selectbox("Select a business type", categories)

    if not selected_type:
        st.info("Please select a business type above.")
        return

    # --- Recommendations ------------------------------------------------
    rec_df = recommend_locations(licences_df, selected_type, top_n=10)
    if rec_df.empty:
        st.warning(f"No data available for '{selected_type}'.")
        return

    # Top-3 recommendation cards
    st.subheader(f"Top 3 Recommended Communities for '{selected_type}'")
    top3 = rec_df.head(3)
    cols = st.columns(3)
    for idx, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
        with col:
            st.markdown(f"### #{idx + 1}: {row['comdistnm']}")
            st.metric("Overall Score", f"{row['overall_score']:.2f}")
            st.metric("Survival Rate", f"{row['type_survival_rate']:.1%}")
            st.metric("Similar Businesses", int(row["type_count"]))
            st.metric("Business Diversity", int(row["business_diversity"]))

    st.divider()

    # Full ranking table
    st.subheader("Full Community Ranking")
    display_cols = [
        "comdistnm", "overall_score", "type_survival_rate", "type_count",
        "total_businesses", "business_diversity", "survival_score",
        "competition_score", "diversity_score",
    ]
    available_display = [c for c in display_cols if c in rec_df.columns]
    st.dataframe(
        rec_df[available_display].rename(columns={
            "comdistnm": "Community",
            "overall_score": "Overall Score",
            "type_survival_rate": "Type Survival Rate",
            "type_count": "Same-Type Businesses",
            "total_businesses": "Total Businesses",
            "business_diversity": "Diversity",
            "survival_score": "Survival Score",
            "competition_score": "Competition Score",
            "diversity_score": "Diversity Score",
        }),
        use_container_width=True,
    )

    st.divider()

    # --- Competition Analysis -------------------------------------------
    st.subheader(f"Competition Analysis: '{selected_type}'")
    comp_df = get_competition_analysis(licences_df, selected_type)
    if not comp_df.empty:
        fig_comp = px.bar(
            comp_df.head(20),
            x="community",
            y="competitor_count",
            color="survival_rate",
            color_continuous_scale="RdYlGn",
            labels={
                "community": "Community",
                "competitor_count": "Number of Competitors",
                "survival_rate": "Survival Rate",
            },
        )
        fig_comp.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- Demographic alignment ------------------------------------------
    st.subheader("Demographic Alignment")
    if not rec_df.empty:
        fig_scatter = px.scatter(
            rec_df,
            x="total_businesses",
            y="type_survival_rate",
            size="business_diversity",
            color="overall_score",
            hover_name="comdistnm",
            color_continuous_scale="Viridis",
            labels={
                "total_businesses": "Total Businesses in Community",
                "type_survival_rate": f"'{selected_type}' Survival Rate",
                "business_diversity": "Category Diversity",
                "overall_score": "Score",
            },
        )
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ===================================================================
# PAGE: Model Performance
# ===================================================================

def page_model_performance() -> None:
    """Render the Model Performance page."""
    st.title("Model Performance")
    st.markdown("Compare classification models trained to predict business survival.")

    # Train models (cached)
    rf_result = _train_rf(licences_df)
    xgb_result = _train_xgb(licences_df)

    # --- Classifier comparison ------------------------------------------
    st.subheader("Classifier Comparison")
    comparison = pd.DataFrame([
        {"Model": "Random Forest", **{k: v for k, v in rf_result["metrics"].items()
                                       if k != "classification_report"}},
        {"Model": "XGBoost", **{k: v for k, v in xgb_result["metrics"].items()
                                 if k != "classification_report"}},
    ])
    st.dataframe(comparison, use_container_width=True)

    # Bar chart of metrics
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_data = comparison.melt(
        id_vars="Model", value_vars=metrics_to_plot,
        var_name="Metric", value_name="Score"
    )
    fig_cmp = px.bar(
        plot_data,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_cmp.update_layout(height=400, yaxis_range=[0, 1])
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.divider()

    # --- Feature importance --------------------------------------------
    st.subheader("Feature Importance")

    col_rf, col_xgb = st.columns(2)
    with col_rf:
        st.markdown("**Random Forest**")
        fi_rf = rf_result["feature_importance"]
        fig_fi_rf = px.bar(
            fi_rf,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Greens",
        )
        fig_fi_rf.update_layout(
            yaxis=dict(autorange="reversed"), height=350, showlegend=False
        )
        st.plotly_chart(fig_fi_rf, use_container_width=True)

    with col_xgb:
        st.markdown("**XGBoost**")
        fi_xgb = xgb_result["feature_importance"]
        fig_fi_xgb = px.bar(
            fi_xgb,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Oranges",
        )
        fig_fi_xgb.update_layout(
            yaxis=dict(autorange="reversed"), height=350, showlegend=False
        )
        st.plotly_chart(fig_fi_xgb, use_container_width=True)

    st.divider()

    # --- Detailed classification reports ---------------------------------
    st.subheader("Detailed Classification Reports")
    with st.expander("Random Forest Classification Report"):
        st.text(rf_result["metrics"]["classification_report"])
    with st.expander("XGBoost Classification Report"):
        st.text(xgb_result["metrics"]["classification_report"])

    # --- Survival model summary -----------------------------------------
    st.divider()
    st.subheader("Survival Model Summary")

    analyzer = SurvivalAnalyzer()
    cox_features = [
        "business_age_days", "event_observed", "is_home_occupation",
        "business_count", "business_diversity",
    ]
    cox_available = [c for c in cox_features if c in licences_df.columns]
    if len(cox_available) >= 3:
        cox_df = licences_df[cox_available].dropna()
        try:
            analyzer.fit_cox(cox_df, duration_col="business_age_days",
                             event_col="event_observed")
            summary = analyzer.get_hazard_summary()
            st.dataframe(summary, use_container_width=True)

            concordance = analyzer.cph.concordance_index_
            st.metric("Cox Model Concordance Index", f"{concordance:.4f}")
        except Exception as exc:
            st.warning(f"Could not fit Cox model: {exc}")
    else:
        st.info("Not enough features for Cox PH summary.")


# ===================================================================
# PAGE: About
# ===================================================================

def page_about() -> None:
    """Render the About page."""
    st.title("About This Project")

    st.markdown("""
    ## Problem Statement

    Small-business failure is a persistent economic challenge. In Calgary,
    thousands of businesses are registered every year, yet a significant
    proportion close within the first few years. Understanding **which
    factors drive business longevity** -- and **where new businesses have
    the best chance of surviving** -- can help entrepreneurs, economic
    development agencies, and municipal planners make better decisions.

    ## Approach

    This project combines **survival analysis** with **machine-learning
    classification** and a **location-recommendation engine** to provide
    actionable insights:

    1. **Survival Analysis (Kaplan-Meier & Cox PH)**
       - Estimates the probability that a business survives past a given
         time horizon (1, 3, 5 years).
       - Identifies community-level and business-level hazard factors.

    2. **Classification (Random Forest & XGBoost)**
       - Predicts whether a business will remain active or close, using
         features such as business age, home-occupation status, community
         density, and diversity.

    3. **Location Recommender**
       - Given a business type, scores every community on survival rate,
         competition level, and demographic fit, then ranks them to
         suggest the best locations.

    ## Dataset

    - **Business Licences** -- 22,000+ records from Calgary Open Data
      (dataset ``vdjc-pybd``), including licence type, issue/expiry
      dates, community district, and status.
    - **Civic Census** -- community-level demographic data
      (dataset ``vsk6-ghca``).

    ## Technology Stack

    | Component | Library |
    |-----------|---------|
    | Survival analysis | lifelines |
    | Classification | scikit-learn, XGBoost |
    | Visualisation | Plotly |
    | Web app | Streamlit |
    | Data access | sodapy (Socrata API) |

    ## How to Run

    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    """)


# ===================================================================
# Router
# ===================================================================

PAGES = {
    "Business Dashboard": page_business_dashboard,
    "Survival Analysis": page_survival_analysis,
    "Location Recommender": page_location_recommender,
    "Model Performance": page_model_performance,
    "About": page_about,
}

PAGES[page]()
