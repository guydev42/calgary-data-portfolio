<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Building%20Permit%20Cost%20Predictor&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Machine%20learning%20regression%20on%20484K%2B%20Calgary%20building%20permits&descSize=16&descAlignY=55&descColor=c8e0ff" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/XGBoost-0.89_R²-blue?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Random_Forest-Ensemble-228B22?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Calgary_Open_Data-Socrata_API-orange?style=for-the-badge" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

> **Problem** -- Construction stakeholders in Calgary -- homeowners, developers, and city planners -- need reliable cost estimates early in the planning process. Budget overruns remain one of the biggest challenges in construction.
>
> **Solution** -- This project uses 484K+ historical building permits and machine learning to predict project costs from permit characteristics, location, and scope.
>
> **Impact** -- Provides data-driven cost estimates that reduce budget uncertainty, helping planners and developers make informed financial decisions before construction begins.

---

## Results

| Metric | XGBoost | Gradient Boosting | Random Forest |
|--------|---------|-------------------|---------------|
| R-squared | **~0.89** | ~0.85 | ~0.82 |
| MAE ($) | ~30,000 | ~32,000 | ~35,000 |
| RMSE ($) | ~80,000 | ~85,000 | ~90,000 |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Calgary Open   │────>│  Data ingestion  │────>│  Feature         │────>│  Model         │────>│  Streamlit      │
│  Data (Socrata) │     │  & cleaning      │     │  engineering     │     │  training      │     │  dashboard      │
│  484K+ permits  │     │  Log-transform   │     │  Community aggs  │     │  XGBoost       │     │  Cost predictor │
│                 │     │  Outlier removal  │     │  Temporal feats  │     │  Ridge/RF/GB   │     │  Community view │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────┘     └─────────────────┘
```

---

## Project structure

<details>
<summary>Click to expand</summary>

```
project_01_building_permit_cost_predictor/
├── app.py                          # Streamlit dashboard
├── index.html                      # Static landing page
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   └── building_permits.csv        # Cached permit data
├── models/
│   ├── best_model.joblib           # Trained XGBoost model
│   ├── feature_names.joblib        # Feature name mapping
│   ├── label_encoders.joblib       # Categorical encoders
│   └── scaler.joblib               # Feature scaler
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py              # Data fetching & feature engineering
    └── model.py                    # Model training & evaluation
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/building-permit-cost-predictor.git
cd building-permit-cost-predictor

# Install dependencies
pip install -r requirements.txt

# Fetch data from Calgary Open Data
python src/data_loader.py

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Calgary Open Data -- Building Permits](https://data.calgary.ca/) |
| Records | 484,000+ |
| Access method | Socrata API (sodapy) |
| Key fields | Permit type, work category, community, estimated cost, floor area |
| Target variable | Estimated project cost (log-transformed) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-Socrata_API-orange?style=flat-square" />
</p>

---

## Methodology

### Data ingestion and cleaning

- Fetched 484,000+ permits from the Calgary Open Data API using sodapy
- Removed records with missing or zero cost values
- Log-transformed the heavily right-skewed cost distribution to normalize the target

### Feature engineering

- Engineered community-level aggregates: average/median cost, permit counts, and category distributions
- Created temporal features from issue dates: year, month, day-of-week, and seasonal indicators
- Encoded categorical variables (permit type, work category, community) with label encoding

### Model training and evaluation

- Trained and compared Ridge Regression, Random Forest, Gradient Boosting, and XGBoost regressors
- Used 80/20 train-test split with cross-validation for hyperparameter selection
- Evaluated on R-squared, MAE, and RMSE metrics; XGBoost achieved the best R-squared of ~0.89

### Interactive dashboard

- Built a Streamlit dashboard with a cost prediction tool and community explorer
- Integrated Plotly visualizations for cost distributions, feature importance, and community comparisons

---

## Acknowledgements

- [City of Calgary Open Data Portal](https://data.calgary.ca/) for providing the building permit dataset
- [Socrata Open Data API](https://dev.socrata.com/) for programmatic data access

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>
