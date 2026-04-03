<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Employee%20Engagement%20Analyzer&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Predict%20disengagement%20and%20optimize%20recognition%20reward%20strategies&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-AUC%200.87-9558B2?style=for-the-badge&logo=lightgbm&logoColor=white"/>
  <img src="https://img.shields.io/badge/Segments-3%20identified-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Savings-$200K-f59e0b?style=for-the-badge"/>
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
  <a href="https://guydev42.github.io/calgary-data-portfolio/case-study-engagement.html">Case study</a>
</p>

</div>

---

## Overview

> **Predicting employee disengagement from recognition program data to drive targeted interventions that save $200K annually in retention costs.**

Employee disengagement costs organizations billions in lost productivity, turnover, and replacement expenses each year. This project builds a predictive model for HR teams at SaaS companies (like Accolad) to identify employees at risk of disengaging, enabling proactive interventions through optimized recognition and reward strategies. Using a dataset of 8,000 employees across 5 departments with 12 months of recognition data, the project compares four classification models and quantifies the business value of early disengagement detection through cost-benefit analysis and SHAP explainability.

```
Problem   →  Blanket engagement programs waste budget and miss at-risk employees
Solution  →  LightGBM model with SHAP explainability ranks employees by disengagement risk
Impact    →  AUC 0.87, 3 engagement segments identified, $200K projected annual savings
```

---

## Key results

| Metric | Value |
|--------|-------|
| Best model | LightGBM (AUC 0.87) |
| Top predictor | Monthly recognition frequency |
| Engagement segments | 3 (At-risk, Moderate, Highly engaged) |
| Projected annual savings | $200,000 |
| Employees analyzed | 8,000 |

**Key findings**

- **Recognition frequency** is the strongest predictor of engagement, with zero-recognition employees disengaging at 3x the rate of those receiving regular recognition
- **Short tenure** (under 12 months) combined with no peer recognition creates the highest-risk employee segment
- **Peer recognition** has a stronger engagement effect than manager-only recognition across all departments
- **Customer Support** department shows the highest disengagement rates, driven by lower recognition frequency and higher absenteeism
- **PTO rewards** correlate with the highest engagement scores, while badge-only recognition shows the weakest effect

---

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Raw data    │───▶│  Feature         │───▶│  Model training     │
│  (8K rows)   │    │  engineering     │    │  (4 models)         │
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
project_24_employee_engagement_analyzer/
├── data/                  # Employee engagement dataset (8,000 employees)
├── src/                   # Data loading, feature engineering, model training
├── models/                # Saved best model and scaler
├── outputs/               # Plots, SHAP values, comparison tables
├── notebooks/             # EDA, feature engineering, modeling notebooks
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
cd calgary-data-portfolio/project_24_employee_engagement_analyzer

# Install dependencies
pip install -r requirements.txt

# Generate dataset
python generate_data.py

# Train models and generate outputs
python -c "
from src.data_loader import load_and_prepare
from src.model import train_and_evaluate
X_train, X_test, y_train, y_test, fn = load_and_prepare('data/employee_engagement.csv')
train_and_evaluate(X_train, X_test, y_train, y_test, fn)
"

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic employee recognition data |
| Records | 8,000 employees |
| Features | 13 (recognition events, rewards, surveys, absenteeism) |
| Target | is_disengaged (binary: engagement_score < 4) |
| Departments | 5 (Engineering, Sales, Marketing, Operations, Customer Support) |
| Time span | 12 months of recognition data |
| Intervention cost | $500 per employee |
| Replacement cost | $15,000 per disengaged employee |

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

- Missing value imputation for survey non-responses
- Feature engineering: tenure groups, recognition balance, absenteeism rate, reward efficiency
- Encoding: one-hot encoding for categorical features (department, role level, reward type)
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
- Feature importance comparison across tree-based models
</details>

<details>
<summary><b>Business impact</b></summary>

- Cost-benefit analysis with threshold optimization
- $500 intervention cost vs. $15,000 replacement cost per employee
- 40% re-engagement success rate assumption
- Projected annual savings of $200,000
</details>

---

## Acknowledgements

Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/). Designed for HR analytics teams at SaaS companies optimizing employee recognition programs.

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
