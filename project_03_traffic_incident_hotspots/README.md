# Project 3: Calgary Traffic Incident Hotspot Analyzer

## Problem Statement

Traffic incidents in Calgary cause delays, economic losses, and safety hazards. Understanding **where** and **when** incidents cluster is essential for city planners, emergency services, and commuters. This project applies spatial clustering and temporal classification to identify high-risk hotspots and predict peak-incident periods from the City of Calgary's real-time traffic incident feed.

## Dataset

| Attribute       | Details                                                                 |
|-----------------|-------------------------------------------------------------------------|
| **Source**       | [Calgary Open Data - Traffic Incidents](https://data.calgary.ca/Transportation-Transit/Traffic-Incidents/35ra-9556) |
| **Dataset ID**   | `35ra-9556`                                                            |
| **Records**      | 59,000+ (growing continuously)                                        |
| **Columns**      | 13 fields                                                             |
| **Update Freq.** | Every 10 minutes                                                      |

Key columns: `start_dt`, `latitude`, `longitude`, `quadrant`, `description`, `count`.

## Methodology

### 1. Spatial Clustering (Unsupervised)

- **DBSCAN** with haversine distance to identify density-based hotspot regions; automatically labels noise points that do not belong to any cluster.
- **KMeans** to partition incidents into a configurable number of geographic zones, providing balanced cluster sizes and clear spatial boundaries.

### 2. Temporal Classification (Supervised)

- **Target variable**: Binary label indicating whether a given hour-location combination exceeds the median incident count (high-incident = 1).
- **Features**: Latitude, longitude, hour (with cyclical encoding), day of week, month, quadrant, weekend flag, rush-hour flag.
- **Models**: Random Forest (200 trees) and Gradient Boosting (150 estimators).
- **Evaluation**: Accuracy, precision, recall, F1-score, and 5-fold cross-validated F1.

### 3. Feature Engineering

- Cyclical sine/cosine encoding for hour, day of week, and month.
- Binary flags for weekends and rush-hour windows (7-9 AM, 4-6 PM).
- Quadrant integer encoding.

## Streamlit Application

The interactive dashboard includes five pages:

1. **Incident Dashboard** - Key metrics (total incidents, quadrants, date range), incidents by quadrant bar chart, and incident description analysis.
2. **Hotspot Map** - Scatter Mapbox visualization of incidents colored by DBSCAN or KMeans cluster, centered on Calgary (51.05, -114.07) with adjustable parameters.
3. **Temporal Analysis** - Incidents by hour, day of week, month, year-over-year trends, and an hour-by-day heatmap.
4. **Model Performance** - Classifier comparison table, metrics bar chart, feature importance, confusion matrix, and classification report.
5. **About** - Methodology explanation and project details.

## Project Structure

```
project_03_traffic_incident_hotspots/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts (joblib)
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── screenshots/            # Application screenshots
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching, caching, preprocessing
    └── model.py            # Clustering and classification models
```

## How to Run

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
cd project_03_traffic_incident_hotspots
pip install -r requirements.txt
```

### Launch the App

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Run the Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Key Findings

- **Peak hours**: Traffic incidents spike during morning (7-9 AM) and afternoon (3-6 PM) rush hours.
- **Weekday dominance**: Weekdays see significantly more incidents than weekends.
- **Spatial hotspots**: Major clusters form along Deerfoot Trail, Crowchild Trail, and the Trans-Canada Highway corridor.
- **Classification**: The Gradient Boosting model achieves strong F1 scores for predicting high-incident periods.

## References

- City of Calgary Open Data Portal: https://data.calgary.ca/
- DBSCAN: Ester, M. et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise."
- scikit-learn documentation: https://scikit-learn.org/
- Streamlit documentation: https://docs.streamlit.io/
- Plotly documentation: https://plotly.com/python/
