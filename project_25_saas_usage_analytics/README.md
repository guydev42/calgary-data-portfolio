<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=SaaS%20Usage%20Analytics&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Product%20analytics%20to%20track%20engagement%2C%20retention%2C%20and%20at-risk%20accounts&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Retention-78%25-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AUC-0.87-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MAU-2.4K-f59e0b?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a> •
  <a href="https://guydev42.github.io/calgary-data-portfolio/case-study-saas.html">Case study</a>
</p>

</div>

---

## Overview

> **Product analytics for a SaaS platform -- tracking user engagement, feature adoption, cohort retention, and usage trends to identify at-risk accounts and optimize product strategy.**

SaaS companies lose revenue when users silently disengage before cancelling. This project builds a predictive analytics system to monitor 3,000 users over 6 months of daily activity, identifying at-risk accounts before they churn. Using engagement metrics, feature adoption patterns, and NPS scores, the project compares four classification models and provides actionable insights through cohort retention analysis, feature adoption tracking, and a product health dashboard.

```
Problem   →  At-risk accounts go undetected until cancellation
Solution  →  Logistic Regression model with engagement signals ranks accounts by churn risk
Impact    →  78% retention, AUC 0.87, 2.4K monthly active users tracked
```

---

## Key results

| Metric | Value |
|--------|-------|
| Best model | Logistic Regression (AUC 0.87) |
| Retention rate | 78% |
| Monthly active users | 2,400 |
| Top churn predictor | Low daily logins (< 3/day) |
| Users analyzed | 3,000 |

**Key findings**

- **Low daily logins** (under 3/day) are the strongest individual churn predictor, with churn rates 4x higher than power users
- **Free-tier users** churn at 3x the rate of Enterprise accounts, driven by low feature adoption and engagement
- **Low feature adoption** (fewer than 4 features) dramatically increases churn probability
- **High support ticket volume** (5+ tickets) indicates frustration and correlates strongly with churn
- **NPS detractors** (score 0-6) churn at significantly higher rates than promoters (9-10)

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Raw data    │───▶│  Feature         │───▶│  Model training     │
│  (3K users)  │    │  engineering     │    │  (4 models)         │
└─────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                      │
                          ┌───────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  SHAP                │───▶│  At-risk account     │
              │  explainability      │    │  analysis            │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Streamlit app       │
                                          │  (5-page dashboard)  │
                                          └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_25_saas_usage_analytics/
├── data/                  # SaaS usage dataset (3,000 users)
├── src/                   # Data loading, feature engineering, model training
├── models/                # Saved best model and scaler
├── outputs/               # Plots, SHAP values, comparison tables
├── notebooks/             # EDA, cohort analysis, churn modeling
├── app.py                 # Streamlit dashboard (5 pages)
├── generate_data.py       # Synthetic data generator
├── requirements.txt       # Python dependencies
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_25_saas_usage_analytics

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python generate_data.py

# Train models and generate outputs
python -c "
from src.data_loader import load_and_prepare
from src.model import train_and_evaluate
X_train, X_test, y_train, y_test, fn = load_and_prepare('data/saas_usage.csv')
train_and_evaluate(X_train, X_test, y_train, y_test, fn)
"

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic SaaS usage data |
| Records | 3,000 users, 6 months of daily activity |
| Features | 14 (engagement, adoption, satisfaction, account) |
| Target | is_churned (binary) |
| Outreach cost | $80 per account |
| Avg annual contract value | $3,600 |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-9558B2?style=for-the-badge&logo=lightgbm&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-FF6F00?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
</p>

---

## Methodology

<details>
<summary><b>Data preparation</b></summary>

- Feature engineering: engagement score, feature adoption rate, activity intensity, support rate
- Encoding: ordinal encoding for plan tier, one-hot encoding for industry and login groups
- Stratified train/test split preserving churn distribution
</details>

<details>
<summary><b>Model comparison</b></summary>

- Logistic Regression, Random Forest, XGBoost, LightGBM
- GridSearchCV hyperparameter tuning with 5-fold cross-validation
- Evaluation on AUC, precision, recall, and F1
</details>

<details>
<summary><b>Explainability</b></summary>

- SHAP values for global and local feature importance
- Force plots and summary plots for individual prediction explanations
</details>

<details>
<summary><b>Product analytics</b></summary>

- Cohort retention analysis by signup month
- Feature adoption tracking across plan tiers
- At-risk account scoring with cost-benefit threshold optimization
- Product health metrics dashboard with NPS, engagement, and adoption tracking
</details>

---

## Acknowledgements

Synthetic SaaS usage dataset generated for product analytics demonstration. Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
