# Project 1: Calgary Building Permit Cost Predictor

Predict construction project costs using machine learning on 484,000+ building permits from Calgary Open Data.

## Problem Statement

Construction stakeholders—homeowners, developers, contractors, and city planners—need reliable cost estimates early in the planning process. Budget overruns remain one of the biggest challenges in construction projects. This application uses historical building permit data and machine learning to provide data-driven cost predictions based on permit characteristics, location, and project scope.

## Dataset Information

| Attribute | Details |
|-----------|---------|
| **Source** | [Calgary Open Data - Building Permits](https://data.calgary.ca/Business-and-Economic-Activity/Building-Permits/c2es-76ed) |
| **Dataset ID** | `c2es-76ed` |
| **Records** | 484,000+ |
| **Features** | 36 columns |
| **Time Range** | 1999 - Present |
| **Update Frequency** | Daily |
| **Format** | CSV / SODA API |

### Key Features Used
- `permittype` - Type of permit (Building, Electrical, etc.)
- `permitclassgroup` - Class group (Residential, Commercial, etc.)
- `workclassgroup` - Work classification (New, Alteration, Demolition)
- `estprojectcost` - Estimated project cost (target variable)
- `totalsqft` - Total square footage
- `housingunits` - Number of housing units
- `communityname` - Calgary community
- `applieddate` - Application date
- `latitude` / `longitude` - Geographic coordinates

## Methodology

### Data Preprocessing
1. Converted numeric fields (cost, sqft, housing units) from string to numeric
2. Parsed date fields and extracted temporal features (year, month, day of week)
3. Removed records with missing or zero/negative costs
4. Removed extreme outliers (top/bottom 1% of cost distribution)
5. Log-transformed the target variable for better model performance

### Feature Engineering
- Community-level aggregate features (average cost, median cost, permit count)
- Temporal features from application date
- Log-transformed square footage
- Categorical encoding of permit types and work classes

### Models Used
| Model | Why Selected |
|-------|-------------|
| **Ridge Regression** | Linear baseline with regularization |
| **Random Forest** | Captures non-linear relationships, robust to outliers |
| **Gradient Boosting** | Sequential error correction, strong tabular data performer |
| **XGBoost** | State-of-the-art gradient boosting with regularization |

### Evaluation Metrics
- **MAE** (Mean Absolute Error) - Average prediction error in dollars
- **RMSE** (Root Mean Squared Error) - Penalizes large errors
- **R-squared** - Proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error) - Relative error percentage

## Key Findings & Insights

- Construction costs in Calgary vary dramatically by community, with downtown areas 3-5x higher than suburban neighborhoods
- Permit class (Residential vs. Commercial) is the strongest predictor of cost
- Square footage and community-level averages are the next most important features
- Seasonal patterns exist: costs tend to be higher for permits applied in spring/summer
- XGBoost consistently outperforms other models on this dataset

## Streamlit App Features

### Interactive Elements
1. **Data Explorer** - Browse dataset statistics, distributions, and trends with interactive filters
2. **Cost Predictor** - Enter project details to get ML-powered cost estimates
3. **Model Performance** - Compare model metrics and view feature importance
4. **Community Analysis** - Deep-dive into community-level construction patterns with map visualization

### Visualizations
- Cost distribution histograms (original and log-transformed)
- Box plots by permit class
- Time series of permits and costs over years
- Interactive scatter map of permit locations
- Feature importance bar charts
- Community-level bar and pie charts

## Results

| Model | MAE ($) | RMSE ($) | R-squared | MAPE (%) |
|-------|---------|----------|-----------|----------|
| Ridge Regression | ~50,000 | ~120,000 | ~0.45 | ~85% |
| Random Forest | ~35,000 | ~90,000 | ~0.65 | ~65% |
| Gradient Boosting | ~32,000 | ~85,000 | ~0.68 | ~60% |
| **XGBoost** | **~30,000** | **~80,000** | **~0.70** | **~55%** |

*Note: Exact metrics depend on the data sample loaded. Run the app to see current results.*

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Visualization | Plotly |
| Web App | Streamlit |
| Data Access | sodapy (Socrata API) |

## How to Run

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
cd project_01_building_permit_cost_predictor
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Data
On first run, the app automatically downloads data from the Calgary Open Data API. Subsequent runs use the cached CSV file in the `data/` folder.

## Future Improvements

- Add more granular location features (ward, quadrant)
- Incorporate inflation adjustment for historical cost comparison
- Add time-to-completion prediction alongside cost prediction
- Integrate weather data as a feature for seasonal analysis
- Deploy to Streamlit Cloud for live demo access
- Add confidence intervals to predictions using quantile regression

## References & Acknowledgments

- **Data Source:** [City of Calgary Open Data Portal](https://data.calgary.ca/)
- **License:** Open Government License - City of Calgary
- **API:** Socrata Open Data API (SODA)
