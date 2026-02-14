# Calgary Open Data: ML/DS Project Portfolio

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006600?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

A comprehensive data science and machine learning portfolio built entirely on [Calgary Open Data](https://data.calgary.ca/). This repository contains **10 end-to-end projects**, each with interactive Streamlit applications, demonstrating a range of ML techniques applied to real-world urban challenges.

---

## Portfolio Overview

| # | Project | Description | Tech Stack | Status |
|---|---------|-------------|------------|--------|
| 1 | [Building Permit Cost Predictor](./project_01_building_permit_cost_predictor/) | Predict construction costs from permit features | XGBoost, Random Forest, Plotly | Complete |
| 2 | [Community Crime Classifier](./project_02_community_crime_classifier/) | Classify and analyze crime patterns across communities | Gradient Boosting, Logistic Regression | Complete |
| 3 | [Traffic Incident Hotspot Analyzer](./project_03_traffic_incident_hotspots/) | Spatial clustering and temporal analysis of traffic incidents | DBSCAN, K-Means, Classification | Complete |
| 4 | [River Flow Forecasting](./project_04_river_flow_forecasting/) | Forecast Bow River levels for flood risk monitoring | LSTM, Prophet, ARIMA | Complete |
| 5 | [Shelter Occupancy Predictor](./project_05_shelter_occupancy_predictor/) | Predict emergency shelter demand | Prophet, XGBoost, Time Series | Complete |
| 6 | [Neighborhood Segmentation](./project_06_neighborhood_segmentation/) | Cluster Calgary communities into livability profiles | K-Means, PCA, Hierarchical | Complete |
| 7 | [Dev Permit Approval Predictor](./project_07_dev_permit_approval_predictor/) | Predict development permit approval likelihood | NLP + Gradient Boosting | Complete |
| 8 | [Solar Energy Forecaster](./project_08_solar_energy_forecaster/) | Forecast solar PV production across city facilities | Seasonal Decomposition, Regression | Complete |
| 9 | [Business Survival Recommender](./project_09_business_survival_recommender/) | Analyze business longevity and recommend locations | Survival Analysis, NLP | Complete |
| 10 | [Water Quality Anomaly Detection](./project_10_water_quality_anomaly_detection/) | Detect anomalies in watershed water quality data | Isolation Forest, Statistical Methods | Complete |

---

## Skills Demonstrated

### Machine Learning Techniques
- **Regression:** Linear, Ridge, Lasso, Random Forest, XGBoost, LightGBM
- **Classification:** Logistic Regression, Decision Trees, Gradient Boosting, SVM
- **Clustering:** K-Means, DBSCAN, Hierarchical Clustering, PCA
- **Time Series:** ARIMA, SARIMA, Prophet, LSTM, Exponential Smoothing
- **NLP:** TF-IDF, Text Classification, Feature Extraction from Text
- **Anomaly Detection:** Isolation Forest, Local Outlier Factor, Statistical Process Control
- **Survival Analysis:** Kaplan-Meier, Cox Proportional Hazards
- **Recommendation Systems:** Content-based Filtering

### Tools & Technologies
- **Languages:** Python 3.10+
- **Data Processing:** pandas, NumPy, SciPy
- **Machine Learning:** scikit-learn, XGBoost, LightGBM, TensorFlow
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Web Applications:** Streamlit
- **Data Access:** Socrata Open Data API (SODA), sodapy
- **Version Control:** Git, GitHub

### Domain Expertise
- Urban Transportation & Traffic Management
- Public Safety & Crime Analysis
- Environmental Monitoring & Water Resources
- Real Estate & Construction Economics
- Social Services & Housing
- Renewable Energy & Sustainability
- Business Analytics & Economic Development
- Demographic Analysis & Urban Planning

---

## Project Highlights

### 1. Building Permit Cost Predictor
Predicts construction project costs using Calgary's 484K+ building permit records dating back to 1999. Features interactive cost estimation, community-level benchmarks, and construction trend analysis. Demonstrates regression techniques with extensive feature engineering.

### 2. Community Crime Pattern Classifier
Analyzes 77K+ crime records across Calgary communities, integrating demographic data to identify patterns and risk factors. Interactive community risk dashboard with temporal trend analysis and crime category prediction.

### 3. Traffic Incident Hotspot Analyzer
Combines spatial clustering with temporal analysis on 60K+ traffic incidents. Identifies dangerous intersections, peak incident times, and predicts incident likelihood. Features animated heatmaps and risk scoring.

### 4. Bow River Flow Forecasting & Flood Risk Monitor
Processes 9.5M+ river flow measurements at 5-minute intervals to forecast water levels. Implements LSTM, Prophet, and ARIMA models with flood threshold alerts. Critical application given Calgary's 2013 flood history.

### 5. Emergency Shelter Occupancy Predictor
Forecasts shelter demand using 83K+ daily occupancy records across Calgary's shelter network. Enables proactive resource allocation with seasonal pattern analysis and capacity utilization tracking.

### 6. Calgary Neighborhood Livability Segmentation
Integrates census, crime, business, and housing data to segment 200+ communities into livability clusters. Interactive community comparison tool with radar charts and PCA-based cluster visualization.

### 7. Development Permit Approval Predictor
Leverages 189K+ development permit records with NLP on project descriptions to predict approval probability. Helps developers assess feasibility before investing in detailed applications.

### 8. Calgary Solar Energy Production Forecaster
Forecasts monthly solar PV output across city facilities. Features site-by-site performance benchmarking, seasonal production patterns, and ROI analysis to support Calgary's renewable energy strategy.

### 9. Business Survival Analyzer & Location Recommender
Applies survival analysis to 22K+ business licenses to identify factors driving business longevity. Recommends optimal community locations based on business type, demographics, and competitive landscape.

### 10. Water Quality Anomaly Detection System
Monitors multi-parameter water quality across Calgary's watershed using anomaly detection. Features real-time alerts, parameter correlation analysis, and historical anomaly tracking for environmental protection.

---

## How to Run

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/guydev42/calgary-data-portfolio.git
   cd calgary-data-portfolio
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install shared dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Individual Projects

Each project is self-contained. Navigate to any project folder and run:

```bash
cd project_01_building_permit_cost_predictor
pip install -r requirements.txt  # Install project-specific dependencies
streamlit run app.py
```

The Streamlit app will open in your browser at `http://localhost:8501`.

### Running Jupyter Notebooks

Each project includes exploratory analysis notebooks:

```bash
cd project_01_building_permit_cost_predictor/notebooks
jupyter notebook
```

---

## About the Data

All data in this portfolio comes from the **[City of Calgary Open Data Portal](https://data.calgary.ca/)**.

- **License:** Open Government License - City of Calgary
- **Attribution:** Contains information licensed under the Open Government License - City of Calgary
- **Terms of Use:** [Calgary Open Data Terms](https://data.calgary.ca/stories/s/Open-Calgary-Terms-of-Use/u45n-7awa)
- **API Access:** Data accessed via the Socrata Open Data API (SODA)

The datasets cover transportation, public safety, environment, demographics, economic activity, and city services. Data is continuously updated by the City of Calgary and reflects real operational records.

---

## Contact & Links

- **GitHub:** [github.com/guydev42](https://github.com/guydev42)

---

## Future Work

- Deploy all Streamlit apps to Streamlit Cloud for live demos
- Add Docker containers for each project for easy deployment
- Integrate real-time data feeds for live dashboards (Traffic Incidents, River Flows)
- Build a unified portfolio dashboard that links all 10 apps
- Add A/B testing framework for model comparison
- Expand to cross-dataset projects combining multiple data sources
- Add automated model retraining pipelines

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
