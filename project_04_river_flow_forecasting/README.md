# Project 4: Bow River Flow Forecasting & Flood Risk Monitor

## Problem Statement

The catastrophic **2013 Calgary floods** caused over $6 billion in damage and displaced more than 100,000 residents. Accurate river-flow forecasting is critical for early-warning systems that can protect lives and infrastructure. This project builds a multi-model forecasting tool using historical Bow River data from the City of Calgary's Open Data portal.

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | [Calgary Open Data Portal](https://data.calgary.ca/) |
| **Dataset ID** | `5fdg-ifgr` (River Levels and Flows) |
| **Total Records** | ~9.5 million |
| **Granularity** | 5-minute intervals |
| **Key Fields** | `timestamp`, `level` (metres), `flow_rate` (m³/s), `station` |
| **Fetch Limit** | 50,000 records by default for development |

## Methodology

### 1. Data Engineering
- Raw 5-minute observations are resampled to **daily mean** values.
- **Rolling averages** (7-day, 30-day) capture short- and medium-term momentum.
- **Lag features** (1, 7, 14, 30 days) encode autoregressive structure.
- Calendar features (day of week, month, year) provide seasonal context.

### 2. ARIMA / SARIMA
- Classical statistical time-series model fitted on the univariate flow series.
- Automatically handles differencing for stationarity.
- Produces well-calibrated confidence intervals.

### 3. Random Forest Regression
- Ensemble of 200 decision trees trained on engineered lag and calendar features.
- Robust to outliers and requires minimal hyperparameter tuning.

### 4. XGBoost Regression
- Gradient-boosted trees (300 rounds, depth 6) for high accuracy on structured tabular data.
- Typically achieves the best MAE and R² among the three approaches.

### 5. Evaluation
- Temporal train/test split (last 20% of data as test set).
- Metrics: **MAE**, **RMSE**, **MAPE**, **R²**.

## Key Findings

- Bow River flow shows strong **seasonal patterns**, peaking in June from snowmelt and receding through winter.
- The **7-day rolling average** is consistently the most important feature, confirming the value of short-term momentum.
- **XGBoost** tends to outperform both Random Forest and ARIMA on this dataset, with lower MAE and higher R².
- ARIMA provides useful confidence intervals that complement point predictions from ML models.

## Project Structure

```
project_04_river_flow_forecasting/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── screenshots/            # App screenshots
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching & feature engineering
    └── model.py            # Model training, evaluation & forecasting
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit app
streamlit run app.py

# 3. (Optional) Run the EDA notebook
jupyter notebook notebooks/01_eda.ipynb
```

On first launch the app will fetch data from the Calgary Open Data API and cache it locally. Subsequent runs load from the cache for faster startup.

## Data Source & Licence

Data provided by the **City of Calgary** under the [Open Data Licence](https://data.calgary.ca/stories/s/Open-Calgary-Terms-of-Use/u45n-7awa).
