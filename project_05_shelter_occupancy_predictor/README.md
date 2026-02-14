# Project 5: Calgary Emergency Shelter Occupancy Predictor

## Problem Statement

Homelessness is a critical social issue in Calgary. Emergency shelters provide essential overnight
accommodation, but unpredictable demand makes resource allocation difficult. When shelters reach
capacity, vulnerable individuals are turned away; when demand is overestimated, resources are
wasted. This project builds a machine learning model to **forecast daily shelter occupancy rates**,
enabling proactive planning, early capacity warnings, and data-driven resource allocation for
Calgary's emergency shelter system.

## Dataset

- **Source**: [City of Calgary Open Data Portal](https://data.calgary.ca/)
- **Dataset**: Emergency Shelters Daily Occupancy
- **Dataset ID**: `7u2t-3wxf`
- **Records**: ~82,869 daily observations
- **Columns** (11):
  | Column | Description |
  |---|---|
  | `date` | Date of observation |
  | `year` | Year |
  | `month` | Month |
  | `city` | City (Calgary) |
  | `sheltertype` | Type of shelter (e.g., emergency, women's) |
  | `sheltername` | Name of the shelter |
  | `organization` | Operating organization |
  | `shelter` | Shelter identifier |
  | `capacity` | Total bed capacity |
  | `overnight` | Number of overnight stays |

## Methodology

1. **Data Collection**: Data fetched from the Socrata API using `sodapy`, with local CSV caching.
2. **Preprocessing**: Date parsing, numeric conversion, occupancy rate computation (`overnight / capacity`).
3. **Feature Engineering**:
   - Temporal: `day_of_week`, `month`, `year`, `day_of_month`
   - Rolling averages: 7-day and 30-day mean occupancy per shelter
   - Lag features: 1-day and 7-day lagged occupancy
   - Categorical encoding: shelter type
4. **Modeling**: Three regression algorithms trained and compared:
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
5. **Evaluation**: Temporal train/test split (80/20) to prevent data leakage. Metrics: MAE, RMSE, R-squared.
6. **Forecasting**: Multi-day-ahead predictions with capacity alerts at the 90% threshold.

## Streamlit Application Features

- **Occupancy Dashboard**: KPIs (total shelters, average occupancy, total capacity), occupancy trends over time, shelter type breakdown.
- **Shelter Analysis**: Individual shelter occupancy trends, month-by-shelter heatmaps, side-by-side shelter comparison.
- **Demand Forecasting**: 7-day and 30-day occupancy predictions, forecast visualization with historical context, capacity alerts when shelters exceed 90%.
- **Model Performance**: Model comparison table, feature importance chart, actual vs predicted scatter plot, residual distribution.
- **About**: Project context, methodology, and data source information.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Project Structure

```
project_05_shelter_occupancy_predictor/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Cached datasets
├── models/                 # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── screenshots/            # App screenshots
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching and preprocessing
    └── model.py            # ML model training and evaluation
```

## Requirements

- Python 3.9+
- pandas, numpy, scikit-learn, xgboost, plotly, streamlit, sodapy, joblib
