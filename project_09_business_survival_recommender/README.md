# Project 9: Calgary Business Survival Analyzer & Location Recommender

## Problem Statement

Small-business failure is one of the most persistent economic challenges faced by cities. In Calgary, thousands of new businesses are registered every year through the municipal licensing system, yet a significant proportion close their doors within the first few years of operation. Understanding which factors drive business longevity and where new businesses have the best chance of surviving can help entrepreneurs choose locations wisely, support economic development agencies in targeting assistance, and enable municipal planners to foster healthier commercial ecosystems.

This project analyses 22,000+ business-licence records from Calgary Open Data to answer three questions:

1. How long do Calgary businesses typically survive, and how does this vary by industry and community?
2. What factors most strongly predict whether a business will remain active or close?
3. Given a business type, which Calgary communities offer the most favourable conditions for success?

## Dataset

| Dataset | Socrata ID | Records | Description |
|---------|-----------|---------|-------------|
| Business Licences | `vdjc-pybd` | 22,000+ | Licence type, issue/expiry dates, community district, status, home-occupation indicator |
| Civic Census | `vsk6-ghca` | varies | Community-level population and demographic data |

Key columns used: `getbusid`, `tradename`, `homeoccind`, `address`, `comdistcd`, `comdistnm`, `licencetypes`, `first_iss_dt`, `exp_dt`, `jobstatusdesc`, `point`, `globalid`.

## Methodology

### 1. Survival Analysis

- **Kaplan-Meier estimator** generates non-parametric survival curves showing the probability of a business remaining active over time, segmented by business type.
- **Cox Proportional-Hazards model** identifies which covariates (home-occupation status, community business density, diversity) significantly increase or decrease the risk of closure.

### 2. Classification

- **Random Forest** and **XGBoost** classifiers predict the binary outcome (survived vs. closed) using engineered features such as business age, community business count, category diversity, and temporal features.
- Models are evaluated on accuracy, precision, recall, F1, and ROC-AUC.

### 3. Location Recommendation

A composite scoring function ranks every Calgary community for a given business type based on:

- **Survival score** (45%) -- historical survival rate for the same business type in the community.
- **Competition score** (30%) -- fewer existing competitors yields a higher score.
- **Diversity score** (25%) -- a more diverse business ecosystem is associated with resilience.

## Project Structure

```
project_09_business_survival_recommender/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── screenshots/            # App screenshots
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching, caching, and feature engineering
    └── model.py            # Survival analysis, classification, and recommendation
```

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   The app will attempt to fetch data from Calgary Open Data on first launch and cache it locally. If the API is unavailable, it falls back to a built-in synthetic dataset for demonstration.

3. (Optional) Set a Socrata app token for higher API rate limits:

   ```bash
   export SOCRATA_APP_TOKEN="your_token_here"
   ```

## Key Findings

- Business survival rates vary significantly by industry and community.
- Home-occupation businesses tend to exhibit different survival patterns compared to commercial-location businesses.
- Communities with higher business diversity generally show better survival rates.
- The location recommender provides actionable guidance by balancing survival history, competition, and ecosystem diversity.
