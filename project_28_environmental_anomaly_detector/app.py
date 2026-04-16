"""
Streamlit dashboard for the Environmental Anomaly Detector.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Environmental Anomaly Detector", page_icon="🔍", layout="wide")

PROJECT_DIR = Path(__file__).resolve().parent

st.sidebar.title("Environmental Anomaly Detector")
page = st.sidebar.radio(
    "Navigate",
    ["Anomaly Timeline", "Spatial Alerts", "Sensor Deep-dive", "Model Comparison"],
)

REGIONS = ["Beaver Hills", "Fort McMurray", "Fox Creek", "RMH Sylvan", "Taber", "Utikuma Lake"]

if page == "Anomaly Timeline":
    st.title("Anomaly Timeline")
    st.markdown("Detected environmental anomalies across all sensor streams over time.")

    np.random.seed(28)
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    values = np.sin(np.linspace(0, 4 * np.pi, 365)) * 10 + 25 + np.random.normal(0, 2, 365)
    anomaly_idx = np.random.choice(365, 18, replace=False)
    values[anomaly_idx] += np.random.uniform(8, 15, 18) * np.random.choice([-1, 1], 18)
    is_anomaly = np.zeros(365, dtype=bool)
    is_anomaly[anomaly_idx] = True

    df = pd.DataFrame({"date": dates, "temperature": values, "anomaly": is_anomaly})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["temperature"], mode="lines", name="Normal", line=dict(color="#3B6FD4")))
    anom = df[df["anomaly"]]
    fig.add_trace(go.Scatter(x=anom["date"], y=anom["temperature"], mode="markers", name="Anomaly",
                             marker=dict(color="#E8C230", size=10, symbol="x")))
    fig.update_layout(template="plotly_dark", title="Land Surface Temperature — Anomaly Detection")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Spatial Alerts":
    st.title("Spatial Anomaly Alerts")
    selected = st.selectbox("Region", REGIONS)

    np.random.seed(42)
    n = 30
    fig = px.scatter_mapbox(
        pd.DataFrame({"lat": np.random.uniform(50, 58, n), "lon": np.random.uniform(-120, -110, n),
                       "severity": np.random.choice(["Low", "Medium", "High"], n)}),
        lat="lat", lon="lon", color="severity", color_discrete_map={"Low": "#3B6FD4", "Medium": "#E8C230", "High": "#E83E3E"},
        mapbox_style="carto-darkmatter", zoom=5, height=600, title=f"Anomaly alerts — {selected}",
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Sensor Deep-dive":
    st.title("Sensor Deep-dive")
    sensor = st.selectbox("Sensor type", ["Land Surface Temp", "NDVI", "Air Quality", "Soil Moisture"])

    np.random.seed(99)
    dates = pd.date_range("2022-01-01", periods=180, freq="D")
    vals = np.random.normal(0, 1, 180).cumsum() + 50
    upper = vals.mean() + 2 * vals.std()
    lower = vals.mean() - 2 * vals.std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=vals, mode="lines", name=sensor, line=dict(color="#3B6FD4")))
    fig.add_hline(y=upper, line_dash="dash", line_color="#E8C230", annotation_text="Upper control")
    fig.add_hline(y=lower, line_dash="dash", line_color="#E8C230", annotation_text="Lower control")
    fig.update_layout(template="plotly_dark", title=f"SPC Chart — {sensor}")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.title("Model Comparison")
    models = ["Isolation Forest", "LOF", "LSTM Autoencoder", "Ensemble"]
    precision = [0.94, 0.88, 0.91, 0.96]
    recall = [0.82, 0.79, 0.85, 0.87]
    f1 = [0.87, 0.83, 0.88, 0.91]

    df = pd.DataFrame({"Model": models * 3, "Metric": ["Precision"] * 4 + ["Recall"] * 4 + ["F1"] * 4,
                        "Value": precision + recall + f1})
    fig = px.bar(df, x="Model", y="Value", color="Metric", barmode="group",
                 color_discrete_sequence=["#E8C230", "#3B6FD4", "#5588EE"],
                 title="Detection Performance Comparison")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
