<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Community%20Crime%20Classifier&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Classification%20of%20Calgary%20communities%20by%20crime%20risk%20level&descSize=16&descAlignY=55&descColor=c8e0ff" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Gradient_Boosting-85%25_Accuracy-blue?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
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

> **Problem** -- Calgary has over 200 communities, each with distinct crime patterns shaped by demographics and location. City planners and residents lack a clear, data-driven way to assess community-level crime risk.
>
> **Solution** -- This project classifies communities by crime risk level (low, medium, high) using 77,000+ crime records and census data, training Gradient Boosting, Random Forest, and Logistic Regression classifiers.
>
> **Impact** -- Helps city planners allocate policing and safety resources and enables residents to make informed decisions about safety and neighbourhood selection.

---

## Results

| Metric | Value |
|--------|-------|
| Best model | Gradient Boosting |
| Accuracy | ~0.85 |
| Weighted F1 | ~0.84 |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Calgary Open   │────>│  Crime stats +   │────>│  Per-capita      │────>│  Classifier    │────>│  Streamlit      │
│  Data (Socrata) │     │  Census data     │     │  rates & risk    │     │  training      │     │  dashboard      │
│  77K+ records   │     │  Merge & clean   │     │  label creation  │     │  Gradient      │     │  Risk explorer  │
│                 │     │                  │     │  Category splits │     │  Boosting      │     │  Community map  │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────┘     └─────────────────┘
```

---

## Project structure

<details>
<summary>Click to expand</summary>

```
project_02_community_crime_classifier/
├── app.py                          # Streamlit dashboard
├── index.html                      # Static landing page
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   ├── census_demographics.csv     # Civic census demographics
│   └── crime_statistics.csv        # Community crime statistics
├── models/                         # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py              # Data fetching & preprocessing
    └── model.py                    # Model training & evaluation
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/community-crime-classifier.git
cd community-crime-classifier

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Calgary Open Data -- Community Crime Statistics](https://data.calgary.ca/) |
| Records | 77,000+ crime records + civic census data |
| Access method | Socrata API (sodapy) |
| Key fields | Community, crime category, crime count, population, demographics |
| Target variable | Risk level (low / medium / high) via percentile thresholds |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-Socrata_API-orange?style=flat-square" />
</p>

---

## Methodology

### Data ingestion and merging

- Fetched crime statistics and civic census data from Calgary Open Data via the Socrata API
- Merged datasets at the community level, linking demographic profiles to crime counts

### Feature engineering

- Aggregated crime counts at the community level with per-capita rates and category breakdowns
- Created risk labels using percentile-based thresholds (low / medium / high)
- Computed demographic ratios and population density features from census data

### Model training and evaluation

- Trained and compared four classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Evaluated with accuracy and F1 scores (weighted and macro)
- Gradient Boosting achieved the best accuracy of ~85% with weighted F1 of ~0.84

### Interactive dashboard

- Built a Streamlit dashboard for exploring community risk levels and crime breakdowns
- Visualizations include community maps, risk distribution charts, and feature importance plots

---

## Acknowledgements

- [City of Calgary Open Data Portal](https://data.calgary.ca/) for providing crime and census datasets
- [Socrata Open Data API](https://dev.socrata.com/) for programmatic data access

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>
