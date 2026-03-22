<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Calgary%20311%20Service%20Request%20Router&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=NLP-powered%20multi-label%20classification%20for%20500K%2B%20service%20requests&descSize=16&descAlignY=55&descColor=c8ddf0" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/status-complete-2ea44f?style=for-the-badge" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" />
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

**Problem** -- Calgary receives over 1.7 million 311 service requests annually covering road maintenance, waste collection, bylaw complaints, and dozens of other categories. Manually routing each request to the correct department introduces delays, misrouting, and increased resolution times.

**Solution** -- This project applies NLP-based TF-IDF feature extraction and multi-label classification on 500,000+ historical service requests to automatically predict the responsible department from request descriptions and metadata, enabling faster and more accurate routing across 15 department classes.

**Impact** -- Achieves an F1 score of 0.79 on department prediction, demonstrating the feasibility of automating 311 request triage and reducing the manual routing burden on city staff.

---

## Results

| Metric | Value |
|--------|-------|
| Best model | Gradient Boosting |
| Accuracy | 0.80 |
| Weighted F1 | **0.79** |
| Department classes | 15 |
| Training records | 500,000+ |

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Calgary Open     │────▶│  Text             │────▶│  TF-IDF           │
│  Data (Socrata)   │     │  Preprocessing    │     │  Vectorization    │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                         ┌──────────────────┐     ┌────────▼─────────┐
                         │  Temporal &        │────▶│  Multi-Label      │
                         │  Community         │     │  Classification   │
                         │  Features          │     └────────┬─────────┘
                         └──────────────────┘              │
                                                  ┌────────▼─────────┐
                                                  │  Streamlit        │
                                                  │  Dashboard        │
                                                  └──────────────────┘
```

---

<details>
<summary><strong>Project structure</strong></summary>

```
project_11_311_service_request_router/
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
    └── model.py            # NLP pipeline & classification models
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/calgary-311-router.git
cd calgary-311-router

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Dataset | Source | Records | Key fields |
|---------|--------|---------|------------|
| 311 service requests | Calgary Open Data (`iahh-g8bj`) | 500,000+ | Request description, department, status, community, timestamp |

---

## Tech stack

<p align="center">
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/TF--IDF-NLP-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-API-blue?style=flat-square" />
</p>

---

## Methodology

1. **Data collection** -- Fetched 500,000+ 311 service request records from Calgary Open Data via Socrata API, including request descriptions, department assignments, timestamps, and community identifiers.
2. **Text preprocessing** -- Cleaned request description text, removed stop words, and normalized tokens for NLP feature extraction.
3. **Feature engineering** -- Applied TF-IDF vectorization to request descriptions, parsed timestamps to extract temporal features (hour, day of week, month), computed resolution times, and encoded community-level service type frequencies.
4. **Model training** -- Trained and compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting classifiers across 15 department classes.
5. **Evaluation** -- Assessed models using accuracy, weighted F1, and macro F1 metrics, with Gradient Boosting achieving the best performance at 0.79 weighted F1.
6. **Dashboard** -- Built a Streamlit application for interactive request routing prediction, department workload analysis, and resolution time visualization.

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
