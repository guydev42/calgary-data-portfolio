"""
Streamlit dashboard for Land Use Segmentation & Clustering.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Land Use Segmentation", page_icon="🗺️", layout="wide")

st.sidebar.title("Land Use Segmentation")
page = st.sidebar.radio("Navigate", ["Cluster Map", "Feature Explorer", "Dimensionality Reduction", "Cluster Diagnostics"])

SEGMENTS = ["Urban", "Agricultural", "Forested", "Wetland", "Industrial", "Grassland", "Mixed-use"]
COLORS = ["#E83E3E", "#E8C230", "#28a745", "#17a2b8", "#6c757d", "#fd7e14", "#3B6FD4"]

if page == "Cluster Map":
    st.title("Land Use Cluster Map")
    np.random.seed(29)
    n = 200
    df = pd.DataFrame({
        "lat": np.random.uniform(50, 58, n), "lon": np.random.uniform(-120, -110, n),
        "segment": np.random.choice(SEGMENTS, n),
    })
    fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="segment",
                            color_discrete_sequence=COLORS,
                            mapbox_style="carto-darkmatter", zoom=4, height=650,
                            title="Land Use Segments — All ODAA Regions")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Feature Explorer":
    st.title("Cluster Feature Profiles")
    np.random.seed(29)
    features = ["Elevation", "Slope", "NDVI", "Water Prox.", "Road Density", "Soil Quality"]
    data = []
    for seg in SEGMENTS:
        vals = np.random.uniform(0.2, 0.9, len(features))
        for f, v in zip(features, vals):
            data.append({"Segment": seg, "Feature": f, "Value": v})
    fig = px.bar(pd.DataFrame(data), x="Feature", y="Value", color="Segment", barmode="group",
                 color_discrete_sequence=COLORS, title="Mean Feature Values by Segment")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Dimensionality Reduction":
    st.title("PCA / t-SNE Projection")
    np.random.seed(29)
    n = 500
    labels = np.random.choice(SEGMENTS, n)
    x = np.random.randn(n) * 3
    y = np.random.randn(n) * 3
    for i, seg in enumerate(SEGMENTS):
        mask = labels == seg
        x[mask] += i * 2
        y[mask] += (i % 3) * 2
    fig = px.scatter(pd.DataFrame({"PC1": x, "PC2": y, "Segment": labels}),
                     x="PC1", y="PC2", color="Segment", color_discrete_sequence=COLORS,
                     title="PCA Projection — 7 Land Use Segments")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.title("Cluster Diagnostics")
    k_range = list(range(2, 12))
    np.random.seed(29)
    inertia = [1000 / k + np.random.normal(0, 5) for k in k_range]
    silhouette = [0.3 + 0.05 * k - 0.005 * k ** 2 + np.random.normal(0, 0.02) for k in k_range]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(pd.DataFrame({"k": k_range, "Inertia": inertia}), x="k", y="Inertia",
                      markers=True, title="Elbow Curve")
        fig.add_vline(x=7, line_dash="dash", line_color="#E8C230", annotation_text="k=7")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(pd.DataFrame({"k": k_range, "Silhouette": silhouette}), x="k", y="Silhouette",
                      markers=True, title="Silhouette Score")
        fig.add_vline(x=7, line_dash="dash", line_color="#E8C230", annotation_text="k=7")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
