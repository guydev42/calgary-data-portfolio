# Project 32 — Site Suitability Recommender

Multi-criteria decision analysis (MCDA) recommender scoring Alberta land parcels for development, agriculture, or conservation suitability.

## Overview

This project builds an interactive site suitability recommender using Analytic Hierarchy Process (AHP) and Weighted Linear Combination on ODAA geospatial data. Users adjust criteria weights in real time and see suitability maps update instantly across six Alberta regions.

## Key Results

| Metric | Value |
|--------|-------|
| NDCG@10 | 0.88 |
| Criteria dimensions | 8 |
| Regions scored | 6 |

## Tech Stack

- **Methods:** MCDA, AHP, Weighted Linear Combination
- **Geospatial:** GeoPandas, Rasterio, Shapely, Folium
- **ML:** scikit-learn (normalization, validation)
- **Visualization:** Plotly, Streamlit

## Methodology

1. Assemble multi-layer geospatial features from ODAA
2. Normalize each criterion to 0-1 scale
3. Implement AHP for pairwise criterion weighting
4. Score grid cells via Weighted Linear Combination
5. Validate recommendations with NDCG ranking metrics
6. Deploy interactive recommender dashboard

## Data Source

[Open Data Areas Alberta](https://www.opendataareas.ca/#data) — geospatial datasets across six regions via Altalis Ltd.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```
