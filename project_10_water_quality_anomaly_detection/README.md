# Project 10: Calgary Water Quality Anomaly Detection System

## Problem Statement

Calgary's watersheds supply drinking water to over 1.4 million residents and sustain critical aquatic ecosystems. Continuous monitoring of water quality is essential to detect contamination events, ensure regulatory compliance, and protect ecological health. Manual review of thousands of lab results across dozens of monitoring sites is impractical -- automated anomaly detection can surface unusual readings in near-real time, enabling faster response and more informed decision-making.

This project builds a **multi-method ensemble anomaly detection system** that identifies unusual water quality measurements across Calgary's watershed monitoring network.

## Dataset

**Watershed Water Quality** -- City of Calgary Open Data Portal

- **Dataset ID:** `y8as-bmzj`
- **Key columns:** `sample_site`, `numeric_result`, `formatted_result`, `result_units`, `latitude_degrees`, `longitude_degrees`, `sample_date`, `parameter`, `site_key`
- **Parameters include:** pH, Temperature, Conductance, Dissolved Oxygen, Turbidity, and many more
- **Spatial coverage:** Multiple monitoring sites across the Bow and Elbow river watersheds

Data is fetched via the Socrata Open Data API (`sodapy`) and cached locally as CSV.

## Methodology

### Feature Engineering

1. **Pivot transformation** -- convert the long-form (one row per measurement) data into a wide-form table with one column per parameter.
2. **Rolling statistics** -- 7-day and 30-day rolling mean and standard deviation per site.
3. **Rate of change** -- first-difference features to capture sudden shifts.
4. **Z-score features** -- per-site normalised scores for contextual anomaly assessment.

### Multi-Method Anomaly Detection

Four complementary detection algorithms are combined into an ensemble:

| Method | Description |
|--------|-------------|
| **Isolation Forest** | Tree-based algorithm that isolates anomalies in fewer random splits. Scales well to high-dimensional data. |
| **Local Outlier Factor (LOF)** | Density-based method that flags points in sparser neighbourhood regions. |
| **Statistical (3-sigma rule)** | Classical threshold: values beyond mean +/- 3 standard deviations. |
| **Z-Score detection** | Per-feature z-score analysis with configurable threshold. |

The **ensemble score** is the mean of the four binary predictions (0/1). A sample is classified as anomalous when the ensemble score meets or exceeds a configurable threshold (default: 0.5, i.e., majority vote).

### Evaluation

- When labelled anomalies are available, precision, recall, and F1 score are computed.
- Otherwise, the contamination parameter controls the expected anomaly proportion for unsupervised methods.

## Project Structure

```
project_10_water_quality_anomaly_detection/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Cached CSV data
├── models/                 # Saved model artefacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── screenshots/            # Dashboard screenshots
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching, caching, and feature engineering
    └── model.py            # Anomaly detection models and ensemble
```

## How to Run

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit dashboard:**

   ```bash
   streamlit run app.py
   ```

   The app will fetch data from the Calgary Open Data portal on first run and cache it locally.

3. **(Optional) Run the EDA notebook:**

   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

## Technology Stack

- Python 3.10+
- Streamlit (interactive dashboard)
- Plotly (interactive charts)
- scikit-learn (Isolation Forest, Local Outlier Factor)
- pandas / NumPy (data wrangling)
- sodapy (Socrata Open Data API client)
- joblib (model serialisation)
