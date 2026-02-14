# Project 2: Calgary Community Crime Pattern Classifier

Classify and analyze crime patterns across 200+ Calgary communities using 77,000+ crime records and demographic data.

## Problem Statement

Residents, businesses, and city planners need to understand crime patterns across Calgary communities to make informed decisions about safety investments, community programs, and resource allocation. This project classifies communities by crime risk level and provides comprehensive trend analysis.

## Dataset Information

| Attribute | Details |
|-----------|---------|
| **Primary Source** | [Community Crime Statistics](https://data.calgary.ca/Health-and-Safety/Community-Crime-Statistics/78gh-n26t) |
| **Secondary Source** | [Civic Census by Age & Gender](https://data.calgary.ca/Demographics/Civic-Census-by-Community-Age-and-Gender/vsk6-ghca) |
| **Crime Records** | 77,000+ |
| **Features** | Community, category, crime count, year, month |
| **Update Frequency** | Monthly |

## Methodology

### Data Preprocessing
1. Cleaned and typed all numeric fields
2. Aggregated crime counts at community level with category breakdowns
3. Integrated census demographics for per-capita crime rates
4. Created risk labels using percentile-based thresholds (Low/Medium/High)

### Models Used
| Model | Purpose |
|-------|---------|
| **Logistic Regression** | Linear baseline for multi-class classification |
| **Decision Tree** | Interpretable rule-based classifier |
| **Random Forest** | Ensemble method for robust classification |
| **Gradient Boosting** | Best performer for tabular classification |

### Evaluation Metrics
- Accuracy, F1-Score (Weighted & Macro)

## Streamlit App Features

1. **Crime Dashboard** — Overview statistics, crime by category and community
2. **Community Risk Map** — Risk classification visualization and community comparison
3. **Trend Analysis** — Temporal trends, seasonal patterns, year-month heatmap
4. **Model Performance** — Classifier comparison and feature importance
5. **About** — Methodology and data source documentation

## How to Run

```bash
cd project_02_community_crime_classifier
pip install -r requirements.txt
streamlit run app.py
```

## References & Acknowledgments

- **Data Source:** [City of Calgary Open Data Portal](https://data.calgary.ca/)
- **License:** Open Government License - City of Calgary
