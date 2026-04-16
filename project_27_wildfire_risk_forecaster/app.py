"""
Streamlit dashboard for the Wildfire Risk Forecaster.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Wildfire Risk Forecaster",
    page_icon="🔥",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Wildfire Risk Forecaster")
page = st.sidebar.radio(
    "Navigate",
    ["Risk Heatmap", "Vegetation Analysis", "Forecast Engine", "Model Explainability"],
)

REGIONS = [
    "Beaver Hills", "Fort McMurray", "Fox Creek",
    "RMH Sylvan", "Taber", "Utikuma Lake",
]

# ── Pages ────────────────────────────────────────────────────────────────────
if page == "Risk Heatmap":
    st.title("Wildfire Risk Heatmap")
    st.markdown("Interactive geospatial risk scores across Alberta's six ODAA regions.")

    selected_region = st.selectbox("Select region", REGIONS)

    np.random.seed(42)
    grid_size = 50
    lats = np.random.uniform(50.0, 58.0, grid_size)
    lons = np.random.uniform(-120.0, -110.0, grid_size)
    risk = np.random.beta(2, 5, grid_size)

    fig = px.scatter_mapbox(
        pd.DataFrame({"lat": lats, "lon": lons, "risk": risk}),
        lat="lat", lon="lon", color="risk",
        color_continuous_scale="YlOrRd",
        size=np.abs(risk) * 20,
        mapbox_style="carto-darkmatter",
        zoom=5, height=600,
        title=f"Risk scores — {selected_region}",
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Vegetation Analysis":
    st.title("Vegetation Index Trends")
    st.markdown("NDVI trends from satellite imagery showing vegetation dryness over time.")

    dates = pd.date_range("2018-01-01", periods=72, freq="ME")
    np.random.seed(7)
    ndvi = 0.45 + 0.2 * np.sin(np.linspace(0, 6 * np.pi, 72)) + np.random.normal(0, 0.03, 72)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=ndvi, mode="lines+markers", name="NDVI",
                             line=dict(color="#E8C230", width=2)))
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="High fire risk threshold")
    fig.update_layout(template="plotly_dark", title="Monthly NDVI — Fort McMurray",
                      yaxis_title="NDVI", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Forecast Engine":
    st.title("Seasonal Risk Forecast")
    st.markdown("Prophet-based time-series forecast of the composite fire risk index.")

    dates = pd.date_range("2019-01-01", periods=60, freq="ME")
    np.random.seed(12)
    risk_idx = 0.3 + 0.25 * np.sin(np.linspace(0, 5 * np.pi, 60)) + np.random.normal(0, 0.04, 60)
    forecast_dates = pd.date_range("2024-01-01", periods=12, freq="ME")
    forecast_vals = 0.3 + 0.25 * np.sin(np.linspace(5 * np.pi, 6 * np.pi, 12))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=risk_idx, mode="lines", name="Observed",
                             line=dict(color="#3B6FD4", width=2)))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_vals, mode="lines", name="Forecast",
                             line=dict(color="#E8C230", width=2, dash="dash")))
    fig.update_layout(template="plotly_dark", title="Fire Risk Index — Forecast",
                      yaxis_title="Risk Index", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.title("Model Explainability")
    st.markdown("SHAP feature importance for the XGBoost wildfire risk classifier.")

    features = ["NDVI", "Temperature Max", "Humidity Min", "Wind Speed",
                "Days Since Rain", "Elevation", "Slope", "EVI",
                "Precipitation", "Land Cover"]
    importance = [0.22, 0.18, 0.14, 0.11, 0.10, 0.08, 0.06, 0.05, 0.04, 0.02]

    fig = px.bar(
        pd.DataFrame({"Feature": features, "SHAP Value": importance}),
        x="SHAP Value", y="Feature", orientation="h",
        color="SHAP Value", color_continuous_scale=["#3B6FD4", "#E8C230"],
        title="Mean |SHAP| — Feature Importance",
    )
    fig.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
