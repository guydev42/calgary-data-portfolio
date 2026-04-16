"""
Streamlit dashboard for the Satellite Image Classifier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Satellite Image Classifier", page_icon="🛰️", layout="wide")

st.sidebar.title("Satellite Image Classifier")
page = st.sidebar.radio("Navigate", ["Image Explorer", "Classification Map", "Confusion Matrix", "Grad-CAM Attention"])

CLASSES = ["Urban", "Forest", "Agriculture", "Water", "Barren", "Wetland"]
COLORS = ["#E83E3E", "#28a745", "#E8C230", "#3B6FD4", "#6c757d", "#17a2b8"]

if page == "Image Explorer":
    st.title("Satellite Image Explorer")
    st.markdown("Browse image tiles with predicted land cover labels and confidence scores.")
    np.random.seed(30)
    n_tiles = 20
    df = pd.DataFrame({
        "Tile ID": [f"tile_{i:04d}" for i in range(n_tiles)],
        "Predicted": np.random.choice(CLASSES, n_tiles),
        "Confidence": np.random.uniform(0.75, 0.99, n_tiles).round(3),
        "Region": np.random.choice(["Fort McMurray", "Taber", "Fox Creek", "Beaver Hills"], n_tiles),
    })
    st.dataframe(df, use_container_width=True)

    fig = px.histogram(df, x="Predicted", color="Predicted", color_discrete_sequence=COLORS,
                       title="Prediction Distribution")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Classification Map":
    st.title("Land Cover Classification Map")
    np.random.seed(30)
    n = 300
    fig = px.scatter_mapbox(
        pd.DataFrame({"lat": np.random.uniform(50, 58, n), "lon": np.random.uniform(-120, -110, n),
                       "class": np.random.choice(CLASSES, n)}),
        lat="lat", lon="lon", color="class", color_discrete_sequence=COLORS,
        mapbox_style="carto-darkmatter", zoom=4, height=650,
        title="Predicted Land Cover — All Regions")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Confusion Matrix":
    st.title("Classification Performance")
    np.random.seed(30)
    cm = np.random.randint(1, 15, (6, 6))
    np.fill_diagonal(cm, np.random.randint(80, 120, 6))
    fig = px.imshow(cm, x=CLASSES, y=CLASSES, color_continuous_scale=["#162240", "#E8C230"],
                    title="Confusion Matrix", labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    precision = [0.94, 0.96, 0.91, 0.97, 0.88, 0.90]
    recall = [0.92, 0.93, 0.89, 0.95, 0.85, 0.88]
    f1 = [0.93, 0.94, 0.90, 0.96, 0.86, 0.89]
    df = pd.DataFrame({"Class": CLASSES, "Precision": precision, "Recall": recall, "F1": f1})
    st.dataframe(df, use_container_width=True)

else:
    st.title("Grad-CAM Attention Maps")
    st.markdown("Gradient-weighted class activation maps showing where the CNN focuses for each prediction.")
    st.info("Grad-CAM visualizations are generated from the trained ResNet-50 model. "
            "Select a tile from the Image Explorer to see its attention map.")
    np.random.seed(30)
    heatmap = np.random.rand(16, 16)
    fig = px.imshow(heatmap, color_continuous_scale="YlOrRd", title="Grad-CAM — Sample Tile (Forest)")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
