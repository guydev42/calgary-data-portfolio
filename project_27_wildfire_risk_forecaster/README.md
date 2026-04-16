# Project 27 — Wildfire Risk Forecaster

Geospatial wildfire risk prediction across Alberta's six ODAA-monitored regions using remote sensing, vegetation indices, and weather data.

## Overview

Alberta faces increasing wildfire threats, particularly around Fort McMurray. This project combines satellite imagery (GeoTIFF), vegetation indices (NDVI/EVI), and historical weather data from the Open Data Areas Alberta (ODAA) platform to build spatial risk models that predict wildfire probability.

## Key Results

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.91 |
| F1-score (high-risk) | 0.87 |
| Regions covered | 6 |

## Tech Stack

- **ML:** XGBoost, Random Forest, Prophet
- **Geospatial:** GeoPandas, Rasterio, Shapely, Folium
- **Visualization:** Plotly, Streamlit
- **Data:** ODAA / Altalis (GeoTIFF, Shapefiles)

## Methodology

1. Ingest GeoTIFF satellite imagery and shapefiles from ODAA via Altalis
2. Compute NDVI/EVI vegetation indices from multispectral bands
3. Merge with weather station data and historical fire occurrence records
4. Train XGBoost and Random Forest classifiers on spatial grid cells
5. Layer Prophet time-series forecasts for seasonal risk projection
6. SHAP analysis for feature importance per region

## Project Structure

```
project_27_wildfire_risk_forecaster/
├── app.py                  # Streamlit dashboard
├── config.yaml             # Project configuration
├── requirements.txt        # Python dependencies
├── data/                   # Raw and processed data
├── models/                 # Trained model artifacts
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── data_loader.py      # Data ingestion and processing
│   └── model.py            # Model training and evaluation
└── tests/
```

## Data Source

[Open Data Areas Alberta](https://www.opendataareas.ca/#data) — free geospatial, environmental, and remote sensing datasets provided by Altalis Ltd. under the ODAA Data User License.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```
