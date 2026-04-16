# Project 30 — Satellite Image Classifier

Deep learning land cover classification from high-resolution Maxar/DigitalGlobe satellite imagery using fine-tuned ResNet-50.

## Overview

This project fine-tunes a ResNet-50 CNN on GeoTIFF image tiles from ODAA to classify satellite patches into six land cover categories: urban, forest, agriculture, water, barren, and wetland. Includes Grad-CAM attention maps for model interpretability.

## Key Results

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 93% |
| Macro F1-score | 0.91 |
| Land cover classes | 6 |

## Tech Stack

- **Deep Learning:** PyTorch, ResNet-50, Transfer Learning, Grad-CAM
- **Geospatial:** Rasterio, GeoPandas
- **Visualization:** Plotly, Streamlit, Matplotlib

## Methodology

1. Extract image tiles from Maxar/DigitalGlobe GeoTIFFs
2. Band selection (RGB + NIR) and pixel normalization
3. Data augmentation: rotations, flips, color jitter
4. Fine-tune pre-trained ResNet-50 with 6-class head
5. Evaluate per-class with confusion matrix and F1-score
6. Generate Grad-CAM attention maps

## Data Source

[Open Data Areas Alberta](https://www.opendataareas.ca/#data) — Maxar/DigitalGlobe satellite imagery via Altalis Ltd.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```
