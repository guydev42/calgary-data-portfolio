# Project 29 — Land Use Segmentation & Clustering

Unsupervised clustering of Alberta's ODAA regions into distinct land-use profiles using geospatial and environmental features.

## Overview

This project segments the landscape across six ODAA-monitored Alberta regions into seven distinct land-use profiles (urban, agricultural, forested, wetland, industrial, grassland, mixed-use) using K-Means clustering on terrain, vegetation, and proximity features.

## Key Results

| Metric | Value |
|--------|-------|
| Silhouette score | 0.72 |
| Optimal clusters | 7 |
| Shapefiles processed | 48 |

## Tech Stack

- **ML:** K-Means, DBSCAN, Hierarchical Clustering, PCA, t-SNE
- **Geospatial:** GeoPandas, Rasterio, Shapely
- **Visualization:** Plotly, Streamlit, Folium

## Methodology

1. Load 48 shapefiles and GeoTIFF rasters from all six ODAA regions
2. Extract per-cell features: elevation, slope, NDVI, land cover, soil type, water proximity, road density
3. Standardize features and reduce dimensionality with PCA
4. Compare K-Means, DBSCAN, and hierarchical agglomerative clustering
5. Select optimal k=7 via silhouette analysis and domain interpretability
6. Profile and map land-use segments

## Data Source

[Open Data Areas Alberta](https://www.opendataareas.ca/#data) — 48 shapefiles and GeoTIFF rasters across six regions.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```
