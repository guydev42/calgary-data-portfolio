"""
Calgary Building Permit Cost Predictor
Streamlit application for predicting construction project costs
using Calgary Open Data building permits dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_or_fetch_data, preprocess_data, engineer_features
from src.model import (
    prepare_model_data, train_models, get_feature_importance,
    save_model, load_model,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
)

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calgary Building Permit Cost Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7B8D;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_data(show_spinner="Loading building permits data...")
def load_data():
    """Load and preprocess building permits data."""
    df = load_or_fetch_data(DATA_DIR, limit=100000)
    df = preprocess_data(df)
    df = engineer_features(df)
    return df


@st.cache_resource(show_spinner="Training ML models...")
def train_all_models(data_hash):
    """Train models and cache results."""
    df = load_data()
    X, y, label_encoders, feature_names = prepare_model_data(df)
    trained_models, results, scaler, X_test, y_test = train_models(X, y)
    # Save best model (XGBoost)
    save_model(
        trained_models["XGBoost"], scaler, label_encoders, feature_names, MODEL_DIR
    )
    return trained_models, results, scaler, label_encoders, feature_names


# ── Main App ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Calgary Building Permit Cost Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Predict construction project costs using machine learning on 484K+ building permits from Calgary Open Data"
    "</p>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Data Explorer", "Cost Predictor", "Model Performance", "Community Analysis", "About"],
)

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Ensure you have internet access to download the dataset, or place 'building_permits.csv' in the data/ folder.")
    st.stop()

# ── Page: Data Explorer ─────────────────────────────────────────────────────
if page == "Data Explorer":
    st.header("Data Explorer")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Permits", f"{len(df):,}")
    with col2:
        st.metric("Avg Cost", f"${df['estprojectcost'].mean():,.0f}")
    with col3:
        st.metric("Median Cost", f"${df['estprojectcost'].median():,.0f}")
    with col4:
        st.metric("Communities", f"{df['communityname'].nunique()}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Cost Distribution", "Trends Over Time", "Raw Data"])

    with tab1:
        col_left, col_right = st.columns(2)

        with col_left:
            fig = px.histogram(
                df, x="estprojectcost", nbins=50,
                title="Distribution of Estimated Project Costs",
                labels={"estprojectcost": "Estimated Cost ($)"},
                color_discrete_sequence=["#667eea"],
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            fig = px.histogram(
                df, x="log_cost", nbins=50,
                title="Distribution of Log-Transformed Costs",
                labels={"log_cost": "Log(Cost + 1)"},
                color_discrete_sequence=["#764ba2"],
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        if "permitclassgroup" in df.columns:
            fig = px.box(
                df, x="permitclassgroup", y="estprojectcost",
                title="Cost Distribution by Permit Class",
                labels={
                    "permitclassgroup": "Permit Class",
                    "estprojectcost": "Estimated Cost ($)",
                },
                color="permitclassgroup",
            )
            fig.update_layout(showlegend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "apply_year" in df.columns:
            yearly = df.groupby("apply_year").agg(
                avg_cost=("estprojectcost", "mean"),
                median_cost=("estprojectcost", "median"),
                permit_count=("estprojectcost", "count"),
            ).reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=yearly["apply_year"], y=yearly["permit_count"],
                    name="Permit Count", marker_color="#667eea", opacity=0.6,
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=yearly["apply_year"], y=yearly["median_cost"],
                    name="Median Cost", line=dict(color="#ff6b6b", width=3),
                ),
                secondary_y=True,
            )
            fig.update_layout(
                title="Permits Issued & Median Cost Over Time",
                xaxis_title="Year", height=450,
            )
            fig.update_yaxes(title_text="Permit Count", secondary_y=False)
            fig.update_yaxes(title_text="Median Cost ($)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            if "workclassgroup" in df.columns:
                yearly_type = df.groupby(
                    ["apply_year", "workclassgroup"]
                )["estprojectcost"].count().reset_index()
                yearly_type.columns = ["Year", "Work Class", "Count"]

                fig = px.area(
                    yearly_type, x="Year", y="Count", color="Work Class",
                    title="Permit Volume by Work Class Over Time",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Raw Data Sample")
        show_n = st.slider("Number of rows to display", 10, 500, 50)
        st.dataframe(df.head(show_n), use_container_width=True)

        csv = df.head(1000).to_csv(index=False)
        st.download_button(
            "Download Sample (1000 rows)",
            csv, "building_permits_sample.csv", "text/csv",
        )

# ── Page: Cost Predictor ────────────────────────────────────────────────────
elif page == "Cost Predictor":
    st.header("Predict Construction Cost")
    st.markdown("Enter project details below to get an estimated construction cost.")

    try:
        trained_models, results, scaler, label_encoders, feature_names = train_all_models(
            str(len(df))
        )
    except Exception as e:
        st.error(f"Error training models: {e}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        permit_types = sorted(df["permittype"].dropna().unique()) if "permittype" in df.columns else ["Building Permit"]
        selected_permit_type = st.selectbox("Permit Type", permit_types)

        permit_classes = sorted(df["permitclassgroup"].dropna().unique()) if "permitclassgroup" in df.columns else ["Residential"]
        selected_class = st.selectbox("Permit Class", permit_classes)

        work_classes = sorted(df["workclassgroup"].dropna().unique()) if "workclassgroup" in df.columns else ["New"]
        selected_work_class = st.selectbox("Work Class", work_classes)

    with col2:
        sqft = st.number_input("Total Square Footage", min_value=0, max_value=500000, value=2000, step=100)
        housing_units = st.number_input("Housing Units", min_value=0, max_value=1000, value=1, step=1)

        communities = sorted(df["communityname"].dropna().unique()) if "communityname" in df.columns else []
        selected_community = st.selectbox("Community", communities) if communities else "UNKNOWN"

    if st.button("Predict Cost", type="primary", use_container_width=True):
        # Build input data
        input_data = {}

        # Encode categorical features using label encoders
        cat_mapping = {
            "permittype": selected_permit_type,
            "permitclass": selected_class,
            "permitclassgroup": selected_class,
            "workclass": selected_work_class,
            "workclassgroup": selected_work_class,
        }
        for col in CATEGORICAL_FEATURES:
            if col in feature_names:
                val = cat_mapping.get(col, "Unknown")
                if col in label_encoders:
                    le = label_encoders[col]
                    if val in le.classes_:
                        input_data[col] = le.transform([val])[0]
                    else:
                        input_data[col] = 0
                else:
                    input_data[col] = 0

        # Numerical features
        num_defaults = {
            "totalsqft": sqft,
            "housingunits": housing_units,
            "apply_year": 2026,
            "apply_month": 1,
            "apply_dayofweek": 2,
            "latitude": df["latitude"].median() if "latitude" in df.columns else 51.05,
            "longitude": df["longitude"].median() if "longitude" in df.columns else -114.07,
        }
        # Community stats
        if "communityname" in df.columns and selected_community:
            comm_data = df[df["communityname"] == selected_community]
            if len(comm_data) > 0:
                num_defaults["community_avg_cost"] = comm_data["estprojectcost"].mean()
                num_defaults["community_median_cost"] = comm_data["estprojectcost"].median()
                num_defaults["community_permit_count"] = len(comm_data)

        for col in NUMERICAL_FEATURES:
            if col in feature_names:
                input_data[col] = num_defaults.get(col, 0)

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])[feature_names]

        # Predict with XGBoost
        model = trained_models["XGBoost"]
        log_prediction = model.predict(input_df)[0]
        prediction = np.expm1(log_prediction)

        st.markdown("---")
        st.subheader("Prediction Results")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Estimated Cost", f"${prediction:,.0f}")
        with col_b:
            # Show range (model uncertainty approximation)
            low = prediction * 0.75
            high = prediction * 1.35
            st.metric("Likely Range", f"${low:,.0f} - ${high:,.0f}")
        with col_c:
            if selected_community and "communityname" in df.columns:
                comm_median = df[df["communityname"] == selected_community]["estprojectcost"].median()
                if pd.notna(comm_median) and comm_median > 0:
                    diff_pct = ((prediction - comm_median) / comm_median) * 100
                    st.metric("vs Community Median", f"{diff_pct:+.1f}%")

        # Show all model predictions
        st.markdown("#### All Model Predictions")
        model_preds = {}
        for name, model_obj in trained_models.items():
            if name == "Ridge Regression":
                pred = np.expm1(model_obj.predict(scaler.transform(input_df))[0])
            else:
                pred = np.expm1(model_obj.predict(input_df)[0])
            model_preds[name] = pred

        pred_df = pd.DataFrame(
            [{"Model": k, "Predicted Cost": f"${v:,.0f}"} for k, v in model_preds.items()]
        )
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ── Page: Model Performance ─────────────────────────────────────────────────
elif page == "Model Performance":
    st.header("Model Performance Comparison")

    try:
        trained_models, results, scaler, label_encoders, feature_names = train_all_models(
            str(len(df))
        )
    except Exception as e:
        st.error(f"Error training models: {e}")
        st.stop()

    # Results table
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(2)
    results_df.columns = ["MAE ($)", "RMSE ($)", "R-squared", "MAPE (%)"]

    st.subheader("Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            results_df.reset_index(),
            x="index", y="R-squared",
            title="R-squared Score by Model",
            labels={"index": "Model", "R-squared": "R-squared"},
            color="R-squared",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            results_df.reset_index(),
            x="index", y="MAE ($)",
            title="Mean Absolute Error by Model",
            labels={"index": "Model", "MAE ($)": "MAE ($)"},
            color="MAE ($)",
            color_continuous_scale="Reds_r",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance (XGBoost)")
    importance = get_feature_importance(
        trained_models["XGBoost"], feature_names, "XGBoost"
    )
    if not importance.empty:
        fig = px.bar(
            importance.head(15),
            x="Importance", y="Feature",
            orientation="h",
            title="Top 15 Most Important Features",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

# ── Page: Community Analysis ─────────────────────────────────────────────────
elif page == "Community Analysis":
    st.header("Community-Level Analysis")

    if "communityname" in df.columns:
        # Top communities by permit count
        community_stats = df.groupby("communityname").agg(
            total_permits=("estprojectcost", "count"),
            avg_cost=("estprojectcost", "mean"),
            median_cost=("estprojectcost", "median"),
            total_value=("estprojectcost", "sum"),
        ).reset_index().sort_values("total_permits", ascending=False)

        tab1, tab2, tab3 = st.tabs(["Top Communities", "Cost Map", "Community Deep Dive"])

        with tab1:
            top_n = st.slider("Show top N communities", 10, 50, 20)

            fig = px.bar(
                community_stats.head(top_n),
                x="communityname", y="total_permits",
                title=f"Top {top_n} Communities by Permit Volume",
                labels={"communityname": "Community", "total_permits": "Total Permits"},
                color="avg_cost",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                community_stats.sort_values("median_cost", ascending=False).head(top_n),
                x="communityname", y="median_cost",
                title=f"Top {top_n} Communities by Median Cost",
                labels={"communityname": "Community", "median_cost": "Median Cost ($)"},
                color="total_permits",
                color_continuous_scale="Plasma",
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if "latitude" in df.columns and "longitude" in df.columns:
                map_df = df.dropna(subset=["latitude", "longitude"]).copy()
                if len(map_df) > 5000:
                    map_df = map_df.sample(5000, random_state=42)

                fig = px.scatter_mapbox(
                    map_df,
                    lat="latitude", lon="longitude",
                    color="estprojectcost",
                    size="estprojectcost",
                    size_max=15,
                    color_continuous_scale="Viridis",
                    mapbox_style="carto-positron",
                    zoom=10,
                    title="Building Permit Costs Across Calgary",
                    hover_data=["communityname", "permittype", "estprojectcost"],
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Location data not available for map visualization.")

        with tab3:
            selected = st.selectbox(
                "Select a Community",
                community_stats["communityname"].tolist(),
            )
            comm_df = df[df["communityname"] == selected]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Permits", f"{len(comm_df):,}")
            with col2:
                st.metric("Avg Cost", f"${comm_df['estprojectcost'].mean():,.0f}")
            with col3:
                st.metric("Median Cost", f"${comm_df['estprojectcost'].median():,.0f}")

            if "apply_year" in comm_df.columns:
                yearly = comm_df.groupby("apply_year")["estprojectcost"].agg(
                    ["count", "median"]
                ).reset_index()
                yearly.columns = ["Year", "Count", "Median Cost"]

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Bar(x=yearly["Year"], y=yearly["Count"], name="Permits", marker_color="#667eea"),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(x=yearly["Year"], y=yearly["Median Cost"], name="Median Cost",
                               line=dict(color="#ff6b6b", width=3)),
                    secondary_y=True,
                )
                fig.update_layout(title=f"{selected} - Permits & Cost Over Time", height=400)
                fig.update_yaxes(title_text="Permit Count", secondary_y=False)
                fig.update_yaxes(title_text="Median Cost ($)", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

            if "permitclassgroup" in comm_df.columns:
                fig = px.pie(
                    comm_df["permitclassgroup"].value_counts().reset_index(),
                    values="count", names="permitclassgroup",
                    title=f"{selected} - Permit Types",
                )
                st.plotly_chart(fig, use_container_width=True)

# ── Page: About ─────────────────────────────────────────────────────────────
elif page == "About":
    st.header("About This Project")

    st.markdown("""
    ### Problem Statement
    Construction stakeholders—homeowners, developers, contractors—need reliable cost
    estimates early in the planning process. This application uses machine learning to
    predict construction project costs based on historical building permit data from
    the City of Calgary.

    ### Dataset
    - **Source:** [Calgary Open Data - Building Permits](https://data.calgary.ca/Business-and-Economic-Activity/Building-Permits/c2es-76ed)
    - **Records:** 484,000+ building permits since 1999
    - **Features:** 36 columns including permit type, work class, location, square footage, and cost

    ### Methodology
    1. **Data Preprocessing:** Cleaned missing values, removed outliers, parsed dates
    2. **Feature Engineering:** Log-transformed costs, created community-level aggregates,
       extracted temporal features
    3. **Models Trained:**
       - Ridge Regression (baseline)
       - Random Forest Regressor
       - Gradient Boosting Regressor
       - XGBoost Regressor (best performer)
    4. **Evaluation:** MAE, RMSE, R-squared, MAPE

    ### Technical Stack
    - **Data Processing:** pandas, NumPy
    - **ML:** scikit-learn, XGBoost
    - **Visualization:** Plotly
    - **Web App:** Streamlit
    - **Data Access:** Socrata API (sodapy)

    ### Data Source & License
    Contains information licensed under the Open Government License - City of Calgary.
    Data accessed from [data.calgary.ca](https://data.calgary.ca/).
    """)

    st.markdown("---")
    st.markdown(
        "Built as part of the "
        "[Calgary Open Data ML/DS Portfolio](https://github.com/guydev42/calgary-data-portfolio)"
    )
