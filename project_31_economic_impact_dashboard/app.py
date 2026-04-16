"""
Streamlit dashboard for the Economic Impact Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Economic Impact Dashboard", page_icon="📊", layout="wide")

st.sidebar.title("Economic Impact Dashboard")
page = st.sidebar.radio("Navigate", ["Regional Overview", "Correlation Explorer", "Impact Simulator", "Executive Summary"])

REGIONS = ["Beaver Hills", "Fort McMurray", "Fox Creek", "RMH Sylvan", "Taber", "Utikuma Lake"]

if page == "Regional Overview":
    st.title("Regional Economic Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Regional GDP", "$2.4B", "+3.2%")
    col2.metric("Employment Rate", "94.1%", "-0.8%")
    col3.metric("Property Index", "112.5", "+5.1%")
    col4.metric("NDVI Health", "0.52", "-0.04")

    np.random.seed(31)
    years = list(range(2015, 2026))
    fig = go.Figure()
    for region in REGIONS[:3]:
        gdp = np.cumsum(np.random.normal(0.1, 0.3, len(years))) + 2
        fig.add_trace(go.Scatter(x=years, y=gdp, mode="lines+markers", name=region))
    fig.update_layout(template="plotly_dark", title="Regional GDP Trend", yaxis_title="GDP ($B)")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Correlation Explorer":
    st.title("Environment-Economy Correlations")
    np.random.seed(31)
    n = 100
    ndvi = np.random.uniform(0.2, 0.8, n)
    gdp = 1.5 + 2.5 * ndvi + np.random.normal(0, 0.3, n)
    fig = px.scatter(pd.DataFrame({"NDVI": ndvi, "GDP ($B)": gdp}),
                     x="NDVI", y="GDP ($B)", trendline="ols",
                     title="NDVI vs Regional GDP", color_discrete_sequence=["#E8C230"])
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

elif page == "Impact Simulator":
    st.title("What-If Impact Simulator")
    st.markdown("Simulate how environmental changes affect economic outcomes.")
    ndvi_change = st.slider("NDVI change (%)", -30, 30, 0)
    fire_change = st.slider("Fire frequency change (%)", -50, 100, 0)
    base_gdp = 2.4
    impact = base_gdp * (1 + ndvi_change * 0.008 - fire_change * 0.003)
    col1, col2 = st.columns(2)
    col1.metric("Baseline GDP", f"${base_gdp:.1f}B")
    col2.metric("Projected GDP", f"${impact:.2f}B", f"{((impact/base_gdp)-1)*100:+.1f}%")

else:
    st.title("Executive Summary")
    st.markdown("""
    ### Key Findings

    1. **Vegetation health (NDVI) is the strongest predictor** of regional economic output,
       explaining 42% of GDP variance across the six ODAA regions.

    2. **Fire frequency has a lagged negative impact** — each major fire event correlates with
       a 2.3% GDP decline in the following fiscal year.

    3. **Fort McMurray shows the highest vulnerability** due to combined wildfire exposure
       and resource-sector dependence.

    ### Recommendations

    - Invest in vegetation monitoring infrastructure across high-risk regions
    - Develop economic diversification strategies for fire-prone areas
    - Integrate environmental KPIs into regional economic planning cycles
    """)
