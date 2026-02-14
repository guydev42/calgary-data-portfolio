# Project 6: Calgary Neighborhood Livability Segmentation

## Problem Statement

Choosing where to live in a large city like Calgary involves weighing dozens of
factors -- safety, population density, economic vitality, housing mix, and more.
This project applies unsupervised machine learning to group Calgary communities
into distinct livability segments so that residents, urban planners, and
policymakers can quickly understand how neighbourhoods compare.

## Data Sources

All data is sourced from the **City of Calgary Open Data Portal** (`data.calgary.ca`)
via the Socrata API.

| Dataset | Resource ID | Key Columns |
|---------|-------------|-------------|
| Civic Census by Age/Gender | `vsk6-ghca` | year, code, age_range, males, females |
| Community Crime Statistics | `78gh-n26t` | community, category, crime_count, year, month |
| Business Licences | `vdjc-pybd` | tradename, comdistnm, licencetypes, jobstatusdesc |
| Building Permits | `c2es-76ed` | communityname, estprojectcost, permitclassgroup |

## Methodology

1. **Data Integration** -- Each dataset is fetched (or loaded from cache) and
   aggregated to the community level.  A unified feature matrix of 10 features
   is constructed by merging on community name.

2. **Feature Engineering** -- Features include total population, median age
   proxy, gender ratio, total crimes, crime rate, business count, business
   diversity, average building cost, permit count, and housing mix.  Missing
   values are imputed with medians; features are z-score standardised.

3. **Clustering** -- KMeans clustering is run for k = 2..10 and the optimal k
   is selected via the silhouette score.  Agglomerative clustering (Ward
   linkage) provides an alternative view.

4. **PCA** -- Principal Component Analysis reduces the 10-D feature space to
   2-D for scatter-plot visualisation and provides loadings to interpret what
   each component represents.

5. **Interactive Dashboard** -- A Streamlit app lets users explore communities,
   inspect cluster profiles, compare neighbourhoods via radar charts, and
   visualise PCA results.

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit dashboard
streamlit run app.py
```

On first run the app will fetch data from the Calgary Open Data API and cache
CSV files in the `data/` directory.

## Project Structure

```
project_06_neighborhood_segmentation/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Cached CSV files (auto-generated)
├── models/                 # Saved models (auto-generated)
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis notebook
├── screenshots/            # App screenshots
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching, caching, feature engineering
    └── model.py            # Clustering, PCA, evaluation, persistence
```

## Tech Stack

Python | pandas | NumPy | scikit-learn | Plotly | Streamlit | sodapy | joblib
