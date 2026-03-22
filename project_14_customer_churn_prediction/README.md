<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Customer%20Churn%20Prediction&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Identify%20at-risk%20telecom%20customers%20and%20quantify%20retention%20ROI&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-AUC%200.713-9558B2?style=for-the-badge&logo=lightgbm&logoColor=white"/>
  <img src="https://img.shields.io/badge/Recall-85%25-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Savings-$141K-f59e0b?style=for-the-badge"/>
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
  <a href="https://guydev42.github.io/calgary-data-portfolio/case-study-churn.html">Case study</a>
</p>

</div>

---

## Overview

> **Predicting telecom customer churn with explainable machine learning to drive targeted retention campaigns that save $141K annually.**

Customer churn costs telecom companies billions in lost revenue each year. This project builds a predictive model to identify customers at high risk of cancelling their service, enabling targeted retention campaigns that maximize ROI. Using a dataset of 5,000 telecom customers with 20 features covering demographics, service subscriptions, billing, and account tenure, the project compares four classification models and quantifies the business value of churn intervention through cost-benefit analysis and SHAP explainability.

```
Problem   →  Mass retention campaigns waste budget on low-risk customers
Solution  →  LightGBM model with SHAP explainability ranks customers by churn risk
Impact    →  85% recall at optimized threshold, $141K projected annual savings
```

---

## Key results

| Metric | Value |
|--------|-------|
| Best model | LightGBM (AUC 0.713) |
| Recall | 85% |
| Projected annual savings | $141,000 |
| Top churn predictor | Month-to-month contracts (3x higher churn) |
| Customers scored | 5,000 |

**Key findings**

- **Month-to-month contracts** are the strongest churn predictor, with churn rates 3x higher than two-year contracts
- **Fiber optic customers** churn more than DSL customers, likely due to higher monthly charges and service expectations
- **Electronic check** payment method is associated with significantly higher churn compared to automatic payment methods
- **Short tenure** (under 12 months) combined with no support services creates the highest-risk customer segment
- The optimal intervention threshold produces substantial monthly net savings by balancing intervention costs against retained revenue

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Raw data    │───▶│  Feature         │───▶│  Model training     │
│  (5K rows)   │    │  engineering     │    │  (4 models)         │
└─────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                      │
                          ┌───────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  SHAP                │───▶│  Cost-benefit        │
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
project_14_customer_churn_prediction/
├── data/                  # Telco churn dataset (5,000 customers)
├── src/                   # Data loading, feature engineering, model training
├── models/                # Saved best model and scaler
├── outputs/               # Plots, SHAP values, comparison tables
├── notebooks/             # Exploratory data analysis notebook
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
cd calgary-data-portfolio/project_14_customer_churn_prediction

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python generate_data.py

# Train models and generate outputs
python -c "
from src.data_loader import load_and_prepare
from src.model import train_and_evaluate
X_train, X_test, y_train, y_test, fn = load_and_prepare('data/telco_churn.csv')
train_and_evaluate(X_train, X_test, y_train, y_test, fn)
"

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic telecom customer data |
| Records | 5,000 customers |
| Features | 20 (demographics, services, billing, tenure) |
| Target | Churn (binary) |
| Intervention cost | $50 per customer |
| Retained revenue | $75/month per saved customer |

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

- Missing value imputation
- Feature engineering: tenure groups, support service count, charges per tenure month
- Encoding: label encoding + one-hot encoding for categorical features
</details>

<details>
<summary><b>Model comparison</b></summary>

- Logistic Regression, Random Forest, XGBoost, LightGBM
- GridSearchCV hyperparameter tuning with cross-validation
- Evaluation on AUC, precision, recall, and F1
</details>

<details>
<summary><b>Explainability</b></summary>

- SHAP values for global and local feature importance
- Force plots and summary plots for individual prediction explanations
</details>

<details>
<summary><b>Business impact</b></summary>

- Cost-benefit analysis with threshold optimization
- $50 intervention cost vs. $75/month retained revenue
- Projected annual savings of $141,000
</details>

---

## Acknowledgements

Dataset inspired by the IBM Telco Customer Churn dataset. Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
