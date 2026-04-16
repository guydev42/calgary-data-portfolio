# Project 31 — Economic Impact Dashboard

Quantifying how environmental changes in Alberta's ODAA regions correlate with economic indicators like GDP, property values, and employment.

## Overview

This project merges environmental datasets (NDVI, fire frequency, land surface temperature) with socioeconomic indicators to model the economic impact of environmental degradation. Features an interactive what-if simulator and auto-generated executive summaries.

## Key Results

| Metric | Value |
|--------|-------|
| R-squared | 0.84 |
| Regions analyzed | 6 |
| KPI indicators | 12 |

## Tech Stack

- **ML:** Ridge Regression, Panel Data Models, Causal Inference
- **Geospatial:** GeoPandas, Shapely
- **Visualization:** Plotly, Streamlit
- **Data:** ODAA + Alberta Government economic data

## Methodology

1. Merge ODAA environmental datasets with Alberta economic indicators
2. Build regional panel regression models
3. Quantify environment-to-economy linkages with causal analysis
4. Create interactive what-if impact simulator
5. Auto-generate executive narrative reports
6. Deploy Streamlit dashboard

## Data Source

[Open Data Areas Alberta](https://www.opendataareas.ca/#data) + [AB Data Partnerships](https://www.abdatapartnerships.ca/)

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```
