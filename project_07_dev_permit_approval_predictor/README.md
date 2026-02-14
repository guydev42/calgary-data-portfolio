# Project 7: Calgary Development Permit Approval Predictor

## Problem Statement

Applying for a development permit in Calgary involves significant time, cost, and uncertainty. Applicants -- homeowners, developers, and architects -- often have little insight into whether their application is likely to be approved or refused. This project leverages historical permit data and Natural Language Processing (NLP) to build a machine-learning model that estimates the probability of permit approval based on the permit description text and structured application attributes.

## Dataset

- **Source:** [Calgary Open Data -- Development Permits](https://data.calgary.ca/Business-and-Financial-Services/Development-Permits/6933-unw5)
- **Socrata Dataset ID:** `6933-unw5`
- **Records:** ~188,653
- **Columns:** 40
- **Time Span:** 1979 -- present

### Key Columns

| Column | Description |
|--------|-------------|
| `permitnum` | Unique permit identifier |
| `applieddate` | Date the application was submitted |
| `statuscurrent` | Current status (Approved, Cancelled, etc.) |
| `applicant` | Name of the applicant |
| `category` | Permit category |
| `description` | Free-text description of the proposed work |
| `proposedusecode` | Proposed use code |
| `proposedusedescription` | Proposed use description |
| `landusedistrict` | Land-use district code |
| `landusedistrictdescription` | Land-use district description |
| `permitteddiscretionary` | Whether the use is permitted or discretionary |
| `communitycode` | Community identifier code |
| `communityname` | Community name |
| `ward` | City ward number |
| `quadrant` | City quadrant (NE, NW, SE, SW) |
| `latitude` / `longitude` | Geographic coordinates |

## Methodology

1. **Data Collection** -- Data is fetched via the Socrata API using `sodapy` and cached locally as a CSV file.
2. **Preprocessing** -- Dates are parsed to extract temporal features (year, month, day of week). A binary target is created: approved (1) vs not approved (0). Description text is cleaned for NLP.
3. **NLP Feature Engineering** -- TF-IDF vectorisation of cleaned descriptions (unigrams + bigrams, 500 features, English stop-words removed).
4. **Categorical Encoding** -- Label encoding of permit category, land-use district, community, quadrant, and permitted/discretionary flag.
5. **Modelling** -- Four classifiers are trained and compared:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
6. **Evaluation** -- Accuracy, Precision, Recall, F1 Score, and AUC-ROC on a stratified 80/20 train-test split.

## Project Structure

```
project_07_dev_permit_approval_predictor/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Cached CSV data
├── models/                 # Saved model artefacts (joblib)
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── screenshots/            # App screenshots
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching and preprocessing
    └── model.py            # Feature engineering, training, evaluation
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Run the EDA notebook
jupyter notebook notebooks/01_eda.ipynb

# 3. Launch the Streamlit app
streamlit run app.py
```

The app will automatically fetch the data from the Calgary Open Data portal on first run and cache it locally. Model training happens on demand and the best model is saved for future predictions.

## Technologies

- **Python 3.9+**
- **pandas / NumPy** -- data manipulation
- **scikit-learn** -- TF-IDF, classifiers, evaluation metrics
- **XGBoost** -- gradient boosting classifier
- **Plotly** -- interactive visualisations
- **Streamlit** -- web application framework
- **sodapy** -- Socrata Open Data API client
- **NLTK** -- supplementary text processing
- **joblib** -- model persistence
