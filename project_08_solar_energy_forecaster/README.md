# Calgary Solar Energy Production Forecaster

## Problem Statement

The City of Calgary has committed to ambitious renewable energy and climate action goals, including sourcing increasing amounts of municipal energy from solar photovoltaic (PV) installations. Accurate forecasting of solar energy production across city-owned facilities is critical for grid planning, budget allocation, and measuring progress toward sustainability targets. This project builds a machine learning-based forecasting system that predicts monthly solar PV output for each municipal solar installation, enabling data-driven energy planning.

## Dataset

Data is sourced from the **Calgary Open Data Portal**:

| Dataset | ID | Description |
|---|---|---|
| Solar PV Production | `iric-4rrc` | Monthly solar PV production (kWh) per facility |
| Solar Sites | `tbsv-89ps` | Facility site information (location, capacity) |

**Key columns (Production):**
- `facility_name` — Name of the municipal facility
- `period` — Month in YYYY-MM format
- `solar_pv_production_kwh` — Monthly energy production in kilowatt-hours

## Methodology

1. **Data Collection**: Fetch and cache data from Calgary Open Data via the Socrata API.
2. **Feature Engineering**:
   - Cyclical encoding of month (sin/cos) to capture seasonal patterns
   - Rolling averages (3, 6, 12 months) per facility
   - Lag features (1, 3, 12 months) for autoregressive signals
   - Year as a feature to capture long-term trends
3. **Modeling**: Three regression models are trained and compared:
   - **Ridge Regression** — regularized linear baseline
   - **Random Forest** — ensemble of decision trees
   - **XGBoost** — gradient boosted trees
4. **Evaluation**: Temporal train/test split; metrics include MAE, RMSE, R-squared, and MAPE.
5. **Forecasting**: Iterative multi-step forecasting where each predicted month feeds into subsequent predictions.

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will launch in your browser with an interactive dashboard for exploring solar production data, analyzing individual facilities, generating forecasts, and reviewing model performance.

### Run the Notebook

Open `notebooks/01_eda.ipynb` in Jupyter to explore the data analysis and visualizations.

### Command-Line Training

```bash
cd src
python model.py
```

This trains all models, prints evaluation metrics, saves the best model, and generates a sample forecast.

## Project Structure

```
project_08_solar_energy_forecaster/
├── app.py                  # Streamlit dashboard application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Cached datasets (auto-generated)
├── models/                 # Saved trained models
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis notebook
├── screenshots/            # App screenshots
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching and feature engineering
    └── model.py            # Model training, evaluation, and forecasting
```
