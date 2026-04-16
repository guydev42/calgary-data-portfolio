# Project 28 — Environmental Anomaly Detector

Unsupervised anomaly detection on ODAA environmental sensor data to flag pollution spikes, deforestation, and abnormal land surface temperature patterns.

## Overview

Industrial activity in Alberta generates environmental signals that need continuous monitoring. This project builds an ensemble anomaly detection system using Isolation Forest, Local Outlier Factor, and LSTM Autoencoders on multivariate environmental time series from the Open Data Areas Alberta platform.

## Key Results

| Metric | Value |
|--------|-------|
| Precision (5% rate) | 94% |
| F1-score | 0.89 |
| Datasets processed | 109 |

## Tech Stack

- **ML:** Isolation Forest, LOF, LSTM Autoencoder, SPC charts
- **Geospatial:** GeoPandas, Rasterio, Shapely
- **Deep Learning:** PyTorch
- **Visualization:** Plotly, Streamlit

## Methodology

1. Ingest raster and vector environmental data from all six ODAA regions
2. Extract land surface temperature, spectral reflectance, and vegetation health time series
3. Apply statistical process control charts as first-pass filter
4. Train Isolation Forest and LOF on multivariate feature space
5. Build LSTM Autoencoder for temporal pattern anomalies
6. Ensemble voting to reduce false positives

## Data Source

[Open Data Areas Alberta](https://www.opendataareas.ca/#data) — 109 datasets across six regions.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```
