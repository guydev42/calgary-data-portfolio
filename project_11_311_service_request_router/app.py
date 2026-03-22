"""
Calgary 311 Service Request Router
Streamlit application for automatically routing 311 service requests
to the correct department using machine learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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
    page_title="Calgary 311 Service Request Router",
    page_icon="📞",
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


@st.cache_data(show_spinner="Loading 311 service requests data...")
def load_data():
    """Load and preprocess 311 service requests data."""
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
    # Save best model (Gradient Boosting)
    save_model(
        trained_models["Gradient Boosting"], scaler, label_encoders, feature_names, MODEL_DIR
    )
    return trained_models, results, scaler, label_encoders, feature_names


# ── Main App ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">Calgary 311 Service Request Router</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    "Automatically route 311 service requests to the correct department using ML on 1.7M+ requests from Calgary Open Data"
    "</p>",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Request Dashboard", "Auto-Router", "Department Analysis", "Model Performance", "About"],
)

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Ensure you have internet access to download the dataset, or place '311_requests.csv' in the data/ folder.")
    st.stop()

# ── Page: Request Dashboard ─────────────────────────────────────────────────
if page == "Request Dashboard":
    st.header("Request Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requests", f"{len(df):,}")
    with col2:
        st.metric("Departments", f"{df['agency_responsible'].nunique()}")
    with col3:
        st.metric("Service Types", f"{df['service_request_type'].nunique()}")
    with col4:
        if "resolution_hours" in df.columns:
            avg_res = df["resolution_hours"].median()
            st.metric("Median Resolution (hrs)", f"{avg_res:,.1f}" if pd.notna(avg_res) else "N/A")
        else:
            st.metric("Communities", f"{df['community'].nunique()}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Department Overview", "Channel Analysis", "Trends"])

    with tab1:
        # Top departments bar chart
        dept_counts = df["agency_responsible"].value_counts().head(15).reset_index()
        dept_counts.columns = ["Department", "Request Count"]

        fig = px.bar(
            dept_counts, x="Request Count", y="Department",
            orientation="h",
            title="Top 15 Departments by Request Volume",
            color="Request Count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "channel" in df.columns:
            col_left, col_right = st.columns(2)

            with col_left:
                channel_counts = df["channel"].value_counts().reset_index()
                channel_counts.columns = ["Channel", "Count"]
                fig = px.pie(
                    channel_counts, values="Count", names="Channel",
                    title="Request Volume by Channel",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                if "resolution_hours" in df.columns:
                    channel_res = df.groupby("channel")["resolution_hours"].median().reset_index()
                    channel_res.columns = ["Channel", "Median Resolution (hrs)"]
                    fig = px.bar(
                        channel_res.sort_values("Median Resolution (hrs)", ascending=False),
                        x="Channel", y="Median Resolution (hrs)",
                        title="Median Resolution Time by Channel",
                        color="Median Resolution (hrs)",
                        color_continuous_scale="Reds",
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "year" in df.columns and "month" in df.columns:
            monthly = df.groupby(["year", "month"]).size().reset_index(name="count")
            monthly["date"] = pd.to_datetime(
                monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01"
            )
            monthly = monthly.sort_values("date")

            fig = px.line(
                monthly, x="date", y="count",
                title="Monthly Request Volume Over Time",
                labels={"date": "Date", "count": "Request Count"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Day of week distribution
        if "day_of_week" in df.columns:
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow_counts = df["day_of_week"].value_counts().sort_index().reset_index()
            dow_counts.columns = ["Day", "Count"]
            dow_counts["Day Name"] = dow_counts["Day"].map(lambda x: dow_names[x] if x < 7 else "Unknown")

            fig = px.bar(
                dow_counts, x="Day Name", y="Count",
                title="Requests by Day of Week",
                color="Count", color_continuous_scale="Viridis",
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ── Page: Auto-Router ───────────────────────────────────────────────────────
elif page == "Auto-Router":
    st.header("Automatic Request Router")
    st.markdown("Enter service request details below to predict the responsible department.")

    try:
        trained_models, results, scaler, label_encoders, feature_names = train_all_models(
            str(len(df))
        )
    except Exception as e:
        st.error(f"Error training models: {e}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        service_types = sorted(df["service_request_type"].dropna().unique())
        selected_service_type = st.selectbox("Service Request Type", service_types)

        channels = sorted(df["channel"].dropna().unique()) if "channel" in df.columns else ["Phone"]
        selected_channel = st.selectbox("Channel", channels)

    with col2:
        communities = sorted(df["community"].dropna().unique()) if "community" in df.columns else []
        selected_community = st.selectbox("Community", communities) if communities else "Unknown"

        wards = sorted(df["ward"].dropna().unique()) if "ward" in df.columns else []
        selected_ward = st.selectbox("Ward", wards) if wards else "Unknown"

    if st.button("Route Request", type="primary", use_container_width=True):
        # Build input data
        input_data = {}

        # Encode categorical features
        for col in CATEGORICAL_FEATURES:
            if col in feature_names:
                val = {"channel": selected_channel, "ward": selected_ward}.get(col, "Unknown")
                if col in label_encoders:
                    le = label_encoders[col]
                    if val in le.classes_:
                        input_data[col] = le.transform([val])[0]
                    else:
                        input_data[col] = 0
                else:
                    input_data[col] = 0

        # Encode service_request_type
        if "service_request_type_encoded" in feature_names:
            if "service_request_type" in label_encoders:
                le = label_encoders["service_request_type"]
                if selected_service_type in le.classes_:
                    input_data["service_request_type_encoded"] = le.transform([selected_service_type])[0]
                else:
                    input_data["service_request_type_encoded"] = 0
            else:
                input_data["service_request_type_encoded"] = 0

        # Numerical features
        num_defaults = {
            "hour": 10,
            "day_of_week": 2,
            "month": 6,
            "year": 2026,
        }
        # Community stats
        if "community" in df.columns and selected_community:
            comm_data = df[df["community"] == selected_community]
            if len(comm_data) > 0:
                num_defaults["community_request_count"] = len(comm_data)
                if "resolution_hours" in comm_data.columns:
                    num_defaults["community_avg_resolution"] = comm_data["resolution_hours"].mean()
                else:
                    num_defaults["community_avg_resolution"] = 0.0
            else:
                num_defaults["community_request_count"] = 0
                num_defaults["community_avg_resolution"] = 0.0

        # Service type frequency
        if "service_request_type" in df.columns:
            type_freq = df["service_request_type"].value_counts()
            num_defaults["service_type_frequency"] = type_freq.get(selected_service_type, 0)

        for col in NUMERICAL_FEATURES:
            if col in feature_names:
                input_data[col] = num_defaults.get(col, 0)

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])[feature_names]

        # Predict with Gradient Boosting
        model = trained_models["Gradient Boosting"]
        prediction_encoded = model.predict(input_df)[0]

        # Decode prediction
        target_le = label_encoders["_target"]
        predicted_department = target_le.inverse_transform([prediction_encoded])[0]

        st.markdown("---")
        st.subheader("Routing Result")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Recommended Department", predicted_department)
        with col_b:
            # Estimated resolution time from department stats
            dept_data = df[df["agency_responsible"] == predicted_department]
            if "resolution_hours" in dept_data.columns and len(dept_data) > 0:
                est_hours = dept_data["resolution_hours"].median()
                st.metric("Est. Resolution Time", f"{est_hours:,.1f} hrs" if pd.notna(est_hours) else "N/A")
            else:
                st.metric("Est. Resolution Time", "N/A")
        with col_c:
            if len(dept_data) > 0:
                st.metric("Dept. Request Volume", f"{len(dept_data):,}")

        # Show all model predictions
        st.markdown("#### All Model Predictions")
        model_preds = {}
        for name, model_obj in trained_models.items():
            if name == "Logistic Regression":
                pred = model_obj.predict(scaler.transform(input_df))[0]
            else:
                pred = model_obj.predict(input_df)[0]
            model_preds[name] = target_le.inverse_transform([pred])[0]

        pred_df = pd.DataFrame(
            [{"Model": k, "Predicted Department": v} for k, v in model_preds.items()]
        )
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

# ── Page: Department Analysis ───────────────────────────────────────────────
elif page == "Department Analysis":
    st.header("Department Analysis")

    tab1, tab2, tab3 = st.tabs(["Workload", "Resolution Time", "Monthly Trends"])

    with tab1:
        dept_workload = df["agency_responsible"].value_counts().reset_index()
        dept_workload.columns = ["Department", "Request Count"]

        fig = px.bar(
            dept_workload, x="Department", y="Request Count",
            title="Department Workload Distribution",
            color="Request Count", color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_tickangle=-45, height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "resolution_hours" in df.columns:
            # Filter to reasonable resolution times for visualization
            res_df = df[df["resolution_hours"].between(0, 720)].copy()  # up to 30 days

            fig = px.box(
                res_df, x="agency_responsible", y="resolution_hours",
                title="Resolution Time Distribution by Department",
                labels={"agency_responsible": "Department", "resolution_hours": "Resolution Hours"},
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            dept_resolution = df.groupby("agency_responsible")["resolution_hours"].agg(
                ["median", "mean", "count"]
            ).reset_index()
            dept_resolution.columns = ["Department", "Median Hours", "Mean Hours", "Total Requests"]
            dept_resolution = dept_resolution.sort_values("Median Hours", ascending=False)
            st.dataframe(dept_resolution.round(1), use_container_width=True, hide_index=True)

    with tab3:
        if "year" in df.columns and "month" in df.columns:
            selected_depts = st.multiselect(
                "Select Departments",
                df["agency_responsible"].unique().tolist(),
                default=df["agency_responsible"].value_counts().head(5).index.tolist(),
            )
            if selected_depts:
                filtered = df[df["agency_responsible"].isin(selected_depts)]
                monthly_dept = filtered.groupby(
                    ["year", "month", "agency_responsible"]
                ).size().reset_index(name="count")
                monthly_dept["date"] = pd.to_datetime(
                    monthly_dept["year"].astype(str) + "-" + monthly_dept["month"].astype(str) + "-01"
                )

                fig = px.line(
                    monthly_dept.sort_values("date"),
                    x="date", y="count", color="agency_responsible",
                    title="Monthly Request Trends by Department",
                    labels={"date": "Date", "count": "Requests", "agency_responsible": "Department"},
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

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
    results_df = results_df.round(4)
    results_df.columns = ["Accuracy", "Weighted F1", "Macro F1"]

    st.subheader("Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            results_df.reset_index(),
            x="index", y="Accuracy",
            title="Accuracy by Model",
            labels={"index": "Model"},
            color="Accuracy",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            results_df.reset_index(),
            x="index", y="Weighted F1",
            title="Weighted F1 Score by Model",
            labels={"index": "Model"},
            color="Weighted F1",
            color_continuous_scale="Greens",
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix heatmap (top 10 classes)
    st.subheader("Confusion Matrix (Top 10 Departments)")
    X, y, le_dict, feat_names = prepare_model_data(df.copy())
    _, X_t, _, y_t = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = trained_models["Gradient Boosting"]
    y_pred = best_model.predict(X_t)

    target_le = label_encoders["_target"]
    top_10_classes = pd.Series(y).value_counts().head(10).index.tolist()
    mask = pd.Series(y_t.values).isin(top_10_classes)
    if mask.any():
        y_t_filtered = y_t.values[mask]
        y_p_filtered = y_pred[mask]
        labels = sorted(set(y_t_filtered) & set(top_10_classes))
        cm = confusion_matrix(y_t_filtered, y_p_filtered, labels=labels)
        label_names = target_le.inverse_transform(labels)

        fig = px.imshow(
            cm, x=label_names, y=label_names,
            text_auto=True, color_continuous_scale="Blues",
            title="Confusion Matrix (Top 10 Departments)",
        )
        fig.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance (Gradient Boosting)")
    importance = get_feature_importance(
        trained_models["Gradient Boosting"], feature_names, "Gradient Boosting"
    )
    if not importance.empty:
        fig = px.bar(
            importance.head(15),
            x="Importance", y="Feature",
            orientation="h",
            title="Top Feature Importances",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

# ── Page: About ─────────────────────────────────────────────────────────────
elif page == "About":
    st.header("About This Project")

    st.markdown("""
    ### Problem Statement
    Calgary receives over 1.7 million 311 service requests covering a wide range of
    civic issues—from road maintenance to waste collection to bylaw complaints. Routing
    each request to the correct department quickly and accurately is critical for
    efficient city operations and citizen satisfaction. This application uses machine
    learning to automatically predict the responsible department based on request details.

    ### Dataset
    - **Source:** [Calgary Open Data - 311 Service Requests](https://data.calgary.ca/Services-and-Information/311-Service-Requests/iahh-g8bj)
    - **Dataset ID:** `iahh-g8bj`
    - **Records:** 1,700,000+ service requests
    - **Features:** Request type, channel, community, ward, timestamps, resolution info

    ### Methodology
    1. **Data Preprocessing:** Parsed timestamps, computed resolution hours, extracted
       temporal features (year, month, day of week, hour), filtered to top 15 departments
    2. **Feature Engineering:** Community-level aggregates (request volume, avg resolution),
       service type frequency, channel encoding
    3. **Models Trained:**
       - Logistic Regression (scaled baseline)
       - Decision Tree Classifier
       - Random Forest Classifier
       - Gradient Boosting Classifier (best performer)
    4. **Evaluation:** Accuracy, Weighted F1, Macro F1

    ### Technical Stack
    - **Data Processing:** pandas, NumPy
    - **ML:** scikit-learn
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
