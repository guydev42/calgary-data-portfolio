"""
Calgary Community Crime Pattern Classifier
Streamlit application for analyzing and classifying crime patterns
across Calgary communities using open data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import (
    load_or_fetch_crime_data, load_or_fetch_census_data,
    preprocess_crime_data, preprocess_census_data,
    create_community_features, create_temporal_crime_data,
)
from src.model import (
    create_risk_labels, prepare_classification_data,
    train_classifiers, get_feature_importance, save_model,
)

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calgary Community Crime Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E3A5F; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #6B7B8D; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_data(show_spinner="Loading crime statistics...")
def load_data():
    """Load and preprocess all datasets."""
    crime_df = load_or_fetch_crime_data(DATA_DIR, limit=100000)
    crime_df = preprocess_crime_data(crime_df)

    try:
        census_df = load_or_fetch_census_data(DATA_DIR, limit=100000)
        census_df = preprocess_census_data(census_df)
    except Exception:
        census_df = pd.DataFrame()

    community_df = create_community_features(crime_df, census_df)
    community_df = create_risk_labels(community_df)
    temporal_df = create_temporal_crime_data(crime_df)

    return crime_df, census_df, community_df, temporal_df


@st.cache_resource(show_spinner="Training classification models...")
def train_all_models(data_hash):
    """Train classifiers and cache results."""
    _, _, community_df, _ = load_data()
    X, y, le, feature_names = prepare_classification_data(community_df)
    trained_models, results, scaler, X_test, y_test = train_classifiers(X, y)
    save_model(trained_models["Gradient Boosting"], scaler, le, feature_names, MODEL_DIR)
    return trained_models, results, scaler, le, feature_names


# ── Main App ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Calgary Community Crime Pattern Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Analyze crime patterns across 200+ Calgary communities using 77K+ crime records</p>',
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Crime Dashboard", "Community Risk Map", "Trend Analysis", "Model Performance", "About"],
)

try:
    crime_df, census_df, community_df, temporal_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ── Page: Crime Dashboard ───────────────────────────────────────────────────
if page == "Crime Dashboard":
    st.header("Crime Statistics Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(crime_df):,}")
    with col2:
        st.metric("Communities", f"{crime_df['community'].nunique()}")
    with col3:
        st.metric("Crime Categories", f"{crime_df['category'].nunique()}")
    with col4:
        st.metric("Years Covered", f"{crime_df['year'].min()}-{crime_df['year'].max()}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["By Category", "By Community", "Raw Data"])

    with tab1:
        category_totals = crime_df.groupby("category")["crime_count"].sum().sort_values(ascending=True)

        fig = px.bar(
            x=category_totals.values, y=category_totals.index,
            orientation="h",
            title="Total Crime Incidents by Category",
            labels={"x": "Total Incidents", "y": "Category"},
            color=category_totals.values,
            color_continuous_scale="Reds",
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Category distribution pie chart
        fig = px.pie(
            values=category_totals.values, names=category_totals.index,
            title="Crime Category Distribution",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        top_n = st.slider("Show top N communities by crime count", 10, 50, 20)
        top_communities = community_df.nlargest(top_n, "total_crimes")

        fig = px.bar(
            top_communities, x="community", y="total_crimes",
            color="risk_level",
            color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
            title=f"Top {top_n} Communities by Total Crime Count",
            labels={"community": "Community", "total_crimes": "Total Crimes"},
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Raw Crime Data")
        show_n = st.slider("Rows to display", 10, 500, 50, key="raw_slider")
        st.dataframe(crime_df.head(show_n), use_container_width=True)
        csv = crime_df.head(1000).to_csv(index=False)
        st.download_button("Download Sample", csv, "crime_stats_sample.csv", "text/csv")

# ── Page: Community Risk Map ─────────────────────────────────────────────────
elif page == "Community Risk Map":
    st.header("Community Crime Risk Classification")

    risk_counts = community_df["risk_level"].value_counts()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Low Risk Communities", risk_counts.get("Low", 0), delta=None)
    with col2:
        st.metric("Medium Risk Communities", risk_counts.get("Medium", 0))
    with col3:
        st.metric("High Risk Communities", risk_counts.get("High", 0))

    st.markdown("---")

    # Risk level distribution
    fig = px.histogram(
        community_df, x="total_crimes", color="risk_level",
        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
        title="Crime Count Distribution by Risk Level",
        labels={"total_crimes": "Total Crime Count", "risk_level": "Risk Level"},
        nbins=30,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Community comparison
    st.subheader("Community Comparison")
    selected_communities = st.multiselect(
        "Select communities to compare",
        community_df["community"].sort_values().tolist(),
        default=community_df.nlargest(3, "total_crimes")["community"].tolist(),
    )

    if selected_communities:
        comp_df = community_df[community_df["community"].isin(selected_communities)]
        crime_cols = [c for c in comp_df.columns if c.startswith("crime_")]

        if crime_cols:
            comp_melted = comp_df.melt(
                id_vars=["community"], value_vars=crime_cols[:10],
                var_name="Crime Type", value_name="Count",
            )
            comp_melted["Crime Type"] = comp_melted["Crime Type"].str.replace("crime_", "").str.replace("_", " ").str.title()

            fig = px.bar(
                comp_melted, x="Crime Type", y="Count", color="community",
                barmode="group",
                title="Crime Category Comparison",
            )
            fig.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig, use_container_width=True)

    # Crime rate per capita
    if "crime_rate_per_1000" in community_df.columns:
        st.subheader("Crime Rate per 1,000 Residents")
        rate_df = community_df.dropna(subset=["crime_rate_per_1000"]).nlargest(20, "crime_rate_per_1000")
        fig = px.bar(
            rate_df, x="community", y="crime_rate_per_1000",
            color="risk_level",
            color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
            title="Top 20 Communities by Crime Rate per 1,000 Residents",
        )
        fig.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig, use_container_width=True)

# ── Page: Trend Analysis ─────────────────────────────────────────────────────
elif page == "Trend Analysis":
    st.header("Crime Trend Analysis")

    # Overall trend
    yearly_total = crime_df.groupby("year")["crime_count"].sum().reset_index()
    fig = px.line(
        yearly_total, x="year", y="crime_count",
        title="Total Crime Incidents Over Time",
        labels={"year": "Year", "crime_count": "Total Incidents"},
        markers=True,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # By category over time
    st.subheader("Crime Trends by Category")
    selected_categories = st.multiselect(
        "Select crime categories",
        crime_df["category"].unique().tolist(),
        default=crime_df.groupby("category")["crime_count"].sum().nlargest(5).index.tolist(),
    )

    if selected_categories:
        filtered = temporal_df[temporal_df["category"].isin(selected_categories)]
        yearly_cat = filtered.groupby(["year", "category"])["crime_count"].sum().reset_index()

        fig = px.line(
            yearly_cat, x="year", y="crime_count", color="category",
            title="Crime Trends by Category",
            labels={"year": "Year", "crime_count": "Incidents", "category": "Category"},
            markers=True,
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # Seasonal patterns
    st.subheader("Seasonal Patterns")
    monthly = crime_df.groupby("month")["crime_count"].sum().reset_index()
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    monthly["month_name"] = monthly["month"].map(month_names)

    fig = px.bar(
        monthly, x="month_name", y="crime_count",
        title="Crime Incidents by Month (All Years)",
        labels={"month_name": "Month", "crime_count": "Total Incidents"},
        color="crime_count", color_continuous_scale="YlOrRd",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Year-over-year heatmap
    st.subheader("Year x Month Heatmap")
    heatmap_data = crime_df.groupby(["year", "month"])["crime_count"].sum().unstack(fill_value=0)
    fig = px.imshow(
        heatmap_data, labels={"x": "Month", "y": "Year", "color": "Incidents"},
        title="Crime Incidents Heatmap (Year x Month)",
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── Page: Model Performance ──────────────────────────────────────────────────
elif page == "Model Performance":
    st.header("Classification Model Performance")

    try:
        trained_models, results, scaler, le, feature_names = train_all_models(str(len(community_df)))
    except Exception as e:
        st.error(f"Error training models: {e}")
        st.stop()

    results_df = pd.DataFrame(results).T.round(4)
    st.subheader("Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            results_df.reset_index(), x="index", y="Accuracy",
            title="Accuracy by Model",
            labels={"index": "Model"},
            color="Accuracy", color_continuous_scale="Greens",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            results_df.reset_index(), x="index", y="F1 (Weighted)",
            title="F1 Score (Weighted) by Model",
            labels={"index": "Model"},
            color="F1 (Weighted)", color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance (Gradient Boosting)")
    importance = get_feature_importance(trained_models["Gradient Boosting"], feature_names)
    if not importance.empty:
        fig = px.bar(
            importance.head(15), x="Importance", y="Feature",
            orientation="h", title="Top 15 Features",
            color="Importance", color_continuous_scale="Purples",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

# ── Page: About ──────────────────────────────────────────────────────────────
elif page == "About":
    st.header("About This Project")
    st.markdown("""
    ### Problem Statement
    Understanding crime patterns across Calgary communities is essential for public
    safety planning, resource allocation, and informed decision-making. This application
    classifies communities by crime risk level and analyzes temporal and categorical
    crime trends.

    ### Datasets
    - **Community Crime Statistics** — 77,000+ records of monthly crime counts by community and category
    - **Civic Census Demographics** — Community-level population data by age and gender

    ### Methodology
    1. Aggregated crime counts by community with category breakdowns
    2. Integrated demographic data for per-capita crime rates
    3. Classified communities into Low/Medium/High risk using percentile-based thresholds
    4. Trained multiple classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
    5. Analyzed temporal trends and seasonal patterns

    ### Data Source
    [Calgary Open Data Portal](https://data.calgary.ca/) — Open Government License
    """)
