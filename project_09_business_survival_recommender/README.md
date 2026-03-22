<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Calgary%20Business%20Survival%20Analyzer&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Survival%20analysis%20and%20location%20recommender%20for%2022K%2B%20businesses&descSize=16&descAlignY=55&descColor=c8ddf0" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/status-complete-2ea44f?style=for-the-badge" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/lifelines-survival-FF6F00?style=for-the-badge" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/streamlit-dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
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

**Problem** -- Thousands of new businesses register in Calgary every year, yet many close within their first few years of operation. Entrepreneurs, economic development agencies, and city planners lack a data-driven way to understand which factors drive business longevity and which communities offer the most favourable conditions for survival.

**Solution** -- This project analyzes 22,000+ business licence records using Kaplan-Meier survival curves and a Cox Proportional-Hazards model to quantify closure risk factors, then combines survival probability with competition density and industry diversity into a composite location scoring function that recommends optimal communities for new businesses.

**Impact** -- Enables data-informed decisions for business placement, potentially reducing early-stage closure rates by directing entrepreneurs toward communities with stronger survival profiles and balanced competitive landscapes.

---

## Results

| Metric | Value |
|--------|-------|
| Cox concordance index | **0.68** |
| XGBoost accuracy | 0.80 |
| XGBoost AUC-ROC | 0.86 |
| Location score weighting | Survival 45%, Competition 30%, Diversity 25% |

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Calgary Open     │────▶│  Feature          │────▶│  Survival         │
│  Data (Socrata)   │     │  Engineering      │     │  Analysis         │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                         ┌──────────────────┐              │
                         │  XGBoost          │◀─────────────┘
                         │  Classification   │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐     ┌──────────────────┐
                         │  Location         │────▶│  Streamlit        │
                         │  Scoring Engine   │     │  Dashboard        │
                         └──────────────────┘     └──────────────────┘
```

---

<details>
<summary><strong>Project structure</strong></summary>

```
project_09_business_survival_recommender/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching & feature engineering
    └── model.py            # Survival models & classification
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/calgary-business-survival.git
cd calgary-business-survival

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Dataset | Source | Records | Key fields |
|---------|--------|---------|------------|
| Business licences | Calgary Open Data | 22,000+ | Licence type, issue date, status, community |
| Civic census | Calgary Open Data | Community-level | Population, household counts |

---

## Tech stack

<p align="center">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=flat-square" />
  <img src="https://img.shields.io/badge/lifelines-survival-FF6F00?style=flat-square" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-API-blue?style=flat-square" />
</p>

---

## Methodology

1. **Data collection** -- Fetched business licence and civic census data from Calgary Open Data via Socrata API.
2. **Feature engineering** -- Computed business age, encoded licence types, built community-level aggregate features including population and household counts.
3. **Survival analysis** -- Fitted Kaplan-Meier survival curves segmented by business type and a Cox Proportional-Hazards model to identify statistically significant closure risk factors (concordance index 0.68).
4. **Classification** -- Trained Random Forest and XGBoost classifiers for binary survived-vs-closed prediction, achieving 0.86 AUC-ROC with XGBoost.
5. **Location recommender** -- Built a composite scoring function weighting survival probability (45%), competition density (30%), and industry diversity (25%) to rank Calgary communities for new business placement.
6. **Dashboard** -- Deployed an interactive Streamlit application with survival curves, risk factor analysis, and community recommendation maps.

---

## Acknowledgements

Data provided by the [City of Calgary Open Data Portal](https://data.calgary.ca/). This project was developed as part of a municipal data analytics portfolio.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>

<p align="center">
  Built by <a href="https://github.com/guydev42">Ola K.</a>
</p>
