"""
Streamlit dashboard for the Site Suitability Recommender.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Site Suitability Recommender", page_icon="📍", layout="wide")

st.sidebar.title("Site Suitability Recommender")
page = st.sidebar.radio("Navigate", ["Criteria Builder", "Suitability Map", "Top-k Sites", "Scenario Comparison"])

CRITERIA = ["Elevation", "Slope", "Soil Quality", "Water Proximity",
            "Road Distance", "Vegetation", "Fire Risk", "Land Cover"]

if page == "Criteria Builder":
    st.title("Criteria Weight Builder")
    st.markdown("Adjust importance weights for each suitability criterion.")
    weights = {}
    for c in CRITERIA:
        weights[c] = st.slider(c, 0.0, 1.0, 0.5, 0.05)

    total = sum(weights.values())
    if total > 0:
        normalized = {k: v / total for k, v in weights.items()}
    else:
        normalized = {k: 1 / len(CRITERIA) for k in CRITERIA}

    fig = px.bar(pd.DataFrame({"Criterion": list(normalized.keys()), "Weight": list(normalized.values())}),
                 x="Weight", y="Criterion", orientation="h",
                 color="Weight", color_continuous_scale=["#3B6FD4", "#E8C230"],
                 title="Normalized Criterion Weights")
    fig.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Suitability Map":
    st.title("Suitability Score Map")
    np.random.seed(32)
    n = 200
    fig = px.scatter_mapbox(
        pd.DataFrame({"lat": np.random.uniform(50, 58, n), "lon": np.random.uniform(-120, -110, n),
                       "score": np.random.beta(5, 2, n)}),
        lat="lat", lon="lon", color="score",
        color_continuous_scale="YlOrRd", size=np.random.uniform(5, 15, n).tolist(),
        mapbox_style="carto-darkmatter", zoom=4, height=650,
        title="Composite Suitability Scores — All Regions")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Top-k Sites":
    st.title("Top Recommended Sites")
    np.random.seed(32)
    k = st.slider("Number of sites", 5, 25, 10)
    df = pd.DataFrame({
        "Rank": range(1, k + 1),
        "Region": np.random.choice(["Beaver Hills", "Taber", "Fox Creek", "Fort McMurray", "RMH Sylvan"], k),
        "Score": sorted(np.random.beta(8, 2, k), reverse=True),
        "Elevation (m)": np.random.randint(500, 1500, k),
        "Water (km)": np.random.uniform(0.5, 15, k).round(1),
        "Fire Risk": np.random.choice(["Low", "Medium", "High"], k),
    })
    st.dataframe(df, use_container_width=True)

else:
    st.title("Scenario Comparison")
    scenarios = ["Agriculture", "Solar Farm", "Conservation", "Residential"]
    np.random.seed(32)
    data = []
    for s in scenarios:
        for c in CRITERIA:
            data.append({"Scenario": s, "Criterion": c, "Weight": np.random.uniform(0.05, 0.3)})
    fig = px.bar(pd.DataFrame(data), x="Criterion", y="Weight", color="Scenario", barmode="group",
                 color_discrete_sequence=["#E8C230", "#3B6FD4", "#28a745", "#E83E3E"],
                 title="Criteria Weights by Use-Case Scenario")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
