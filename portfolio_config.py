"""
Portfolio configuration — metadata for all 13 Calgary Open Data ML/DS projects.
"""

# ── GitHub ───────────────────────────────────────────────────────────────────
GITHUB_USER = "guydev42"
GITHUB_REPO = "calgary-data-portfolio"
GITHUB_BASE = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}"

# ── Color palette (matches existing project apps) ───────────────────────────
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "dark": "#1E3A5F",
    "mid": "#6B7B8D",
    "accent": "#ff6b6b",
    "bg_light": "#f8f9fc",
    "card_bg": "#ffffff",
    "success": "#28a745",
    "warning": "#ffc107",
}

# ── Category definitions with badge colours ─────────────────────────────────
CATEGORIES = {
    "Regression": {"color": "#667eea", "icon": "📈"},
    "Classification": {"color": "#764ba2", "icon": "🏷️"},
    "Clustering": {"color": "#28a745", "icon": "🔬"},
    "Time Series": {"color": "#ff6b6b", "icon": "⏳"},
    "NLP": {"color": "#fd7e14", "icon": "📝"},
    "Anomaly Detection": {"color": "#20c997", "icon": "🔍"},
    "Survival Analysis": {"color": "#6f42c1", "icon": "📊"},
    "Recommendation": {"color": "#e83e8c", "icon": "💡"},
    "Spatial Analysis": {"color": "#17a2b8", "icon": "🗺️"},
    "Dimensionality Reduction": {"color": "#6c757d", "icon": "📐"},
}

# ── Project entries ─────────────────────────────────────────────────────────
PROJECTS = [
    {
        "number": 1,
        "title": "Building Permit Cost Predictor",
        "tagline": "Predict construction costs from permit features",
        "description": (
            "Predicts construction project costs using Calgary's 484K+ building "
            "permit records dating back to 1999. Features interactive cost "
            "estimation, community-level benchmarks, and construction trend "
            "analysis. Demonstrates regression techniques with extensive feature "
            "engineering."
        ),
        "categories": ["Regression"],
        "tech_stack": ["XGBoost", "Random Forest", "Ridge Regression", "Plotly", "Streamlit"],
        "dataset": "Building Permits",
        "dataset_size": "484K+ records",
        "key_metric": "R-squared with XGBoost",
        "folder": "project_01_building_permit_cost_predictor",
        "streamlit_url": "",
        "pages": [
            "Data Explorer",
            "Cost Predictor",
            "Model Performance",
            "Community Analysis",
            "About",
        ],
        "methodology": [
            "Data cleaning & outlier removal on 484K permits",
            "Feature engineering: log costs, community aggregates, temporal features",
            "Trained Ridge, Random Forest, Gradient Boosting, XGBoost",
            "Evaluated with MAE, RMSE, R-squared, MAPE",
        ],
    },
    {
        "number": 2,
        "title": "Community Crime Classifier",
        "tagline": "Classify and analyze crime patterns across communities",
        "description": (
            "Analyzes 77K+ crime records across Calgary communities, integrating "
            "demographic data to identify patterns and risk factors. Interactive "
            "community risk dashboard with temporal trend analysis and crime "
            "category prediction."
        ),
        "categories": ["Classification"],
        "tech_stack": ["Gradient Boosting", "Logistic Regression", "Plotly", "Streamlit"],
        "dataset": "Community Crime Statistics",
        "dataset_size": "77K+ records",
        "key_metric": "Multi-class classification accuracy",
        "folder": "project_02_community_crime_classifier",
        "streamlit_url": "",
        "pages": [
            "Crime Dashboard",
            "Community Risk Map",
            "Trend Analysis",
            "Model Performance",
            "About",
        ],
        "methodology": [
            "Merged crime records with community demographic profiles",
            "Temporal feature extraction (year, month, day-of-week)",
            "Trained Logistic Regression, Decision Tree, Gradient Boosting",
            "Community-level risk scoring and mapping",
        ],
    },
    {
        "number": 3,
        "title": "Traffic Incident Hotspot Analyzer",
        "tagline": "Spatial clustering and temporal analysis of traffic incidents",
        "description": (
            "Combines spatial clustering with temporal analysis on 60K+ traffic "
            "incidents. Identifies dangerous intersections, peak incident times, "
            "and predicts incident likelihood. Features animated heatmaps and "
            "risk scoring."
        ),
        "categories": ["Clustering", "Classification", "Spatial Analysis"],
        "tech_stack": ["DBSCAN", "K-Means", "Classification", "Plotly", "Streamlit"],
        "dataset": "Traffic Incidents",
        "dataset_size": "60K+ records",
        "key_metric": "Hotspot cluster identification",
        "folder": "project_03_traffic_incident_hotspots",
        "streamlit_url": "",
        "pages": [
            "Incident Dashboard",
            "Hotspot Map",
            "Temporal Analysis",
            "Model Performance",
            "About",
        ],
        "methodology": [
            "Geocoded traffic incidents across the city",
            "DBSCAN spatial clustering for hotspot identification",
            "K-Means clustering for incident pattern grouping",
            "Classification model for incident severity prediction",
        ],
    },
    {
        "number": 4,
        "title": "River Flow Forecasting",
        "tagline": "Forecast Bow River levels for flood risk monitoring",
        "description": (
            "Processes 9.5M+ river flow measurements at 5-minute intervals to "
            "forecast water levels. Implements LSTM, Prophet, and ARIMA models "
            "with flood threshold alerts. Critical application given Calgary's "
            "2013 flood history."
        ),
        "categories": ["Time Series"],
        "tech_stack": ["LSTM", "Prophet", "ARIMA", "TensorFlow", "Plotly", "Streamlit"],
        "dataset": "River Flow Rates",
        "dataset_size": "9.5M+ records",
        "key_metric": "Multi-step forecast accuracy",
        "folder": "project_04_river_flow_forecasting",
        "streamlit_url": "",
        "pages": [
            "River Dashboard",
            "Flow Forecasting",
            "Seasonal Analysis",
            "Model Performance",
            "About",
        ],
        "methodology": [
            "Resampled 5-minute intervals to hourly/daily aggregates",
            "Seasonal decomposition and stationarity testing",
            "Trained ARIMA, Prophet, and LSTM (TensorFlow) models",
            "Flood threshold alerting with configurable levels",
        ],
    },
    {
        "number": 5,
        "title": "Shelter Occupancy Predictor",
        "tagline": "Predict emergency shelter demand",
        "description": (
            "Forecasts shelter demand using 83K+ daily occupancy records across "
            "Calgary's shelter network. Enables proactive resource allocation "
            "with seasonal pattern analysis and capacity utilization tracking."
        ),
        "categories": ["Time Series", "Regression"],
        "tech_stack": ["Prophet", "XGBoost", "Time Series", "Plotly", "Streamlit"],
        "dataset": "Shelter Occupancy",
        "dataset_size": "83K+ records",
        "key_metric": "Occupancy forecast MAPE",
        "folder": "project_05_shelter_occupancy_predictor",
        "streamlit_url": "",
        "pages": [
            "Occupancy Dashboard",
            "Shelter Analysis",
            "Demand Forecasting",
            "Model Performance",
            "About",
        ],
        "methodology": [
            "Aggregated daily occupancy by shelter and sector",
            "Seasonal pattern analysis and holiday effects",
            "Trained Prophet and XGBoost time-series models",
            "Capacity utilization and resource allocation insights",
        ],
    },
    {
        "number": 6,
        "title": "Neighborhood Segmentation",
        "tagline": "Cluster Calgary communities into livability profiles",
        "description": (
            "Integrates census, crime, business, and housing data to segment "
            "200+ communities into livability clusters. Interactive community "
            "comparison tool with radar charts and PCA-based cluster "
            "visualization."
        ),
        "categories": ["Clustering", "Dimensionality Reduction"],
        "tech_stack": ["K-Means", "PCA", "Hierarchical Clustering", "Plotly", "Streamlit"],
        "dataset": "Census & Community Data",
        "dataset_size": "200+ communities",
        "key_metric": "Silhouette score for cluster quality",
        "folder": "project_06_neighborhood_segmentation",
        "streamlit_url": "",
        "pages": [
            "Community Explorer",
            "Cluster Analysis",
            "Community Comparison",
            "PCA Visualization",
            "About",
        ],
        "methodology": [
            "Integrated census, crime, business, and housing datasets",
            "Feature scaling and PCA for dimensionality reduction",
            "K-Means and Hierarchical clustering with silhouette analysis",
            "Radar-chart community comparison tool",
        ],
    },
    {
        "number": 7,
        "title": "Dev Permit Approval Predictor",
        "tagline": "Predict development permit approval likelihood",
        "description": (
            "Uses 189K+ development permit records with NLP on project "
            "descriptions to predict approval probability. Helps developers "
            "assess feasibility before investing in detailed applications."
        ),
        "categories": ["Classification", "NLP"],
        "tech_stack": ["TF-IDF", "Gradient Boosting", "NLP", "Plotly", "Streamlit"],
        "dataset": "Development Permits",
        "dataset_size": "189K+ records",
        "key_metric": "Approval prediction accuracy",
        "folder": "project_07_dev_permit_approval_predictor",
        "streamlit_url": "",
        "pages": [
            "Permit Dashboard",
            "Approval Predictor",
            "NLP Insights",
            "Model Performance",
            "About",
        ],
        "methodology": [
            "Parsed permit descriptions with TF-IDF vectorization",
            "Combined text features with structured permit attributes",
            "Trained Logistic Regression and Gradient Boosting classifiers",
            "Feature importance analysis on NLP vs structured features",
        ],
    },
    {
        "number": 8,
        "title": "Solar Energy Forecaster",
        "tagline": "Forecast solar PV production across city facilities",
        "description": (
            "Forecasts monthly solar PV output across city facilities. Features "
            "site-by-site performance benchmarking, seasonal production patterns, "
            "and ROI analysis to support Calgary's renewable energy strategy."
        ),
        "categories": ["Time Series", "Regression"],
        "tech_stack": ["Seasonal Decomposition", "Regression", "Plotly", "Streamlit"],
        "dataset": "Solar Energy Production",
        "dataset_size": "City facility records",
        "key_metric": "Monthly production forecast accuracy",
        "folder": "project_08_solar_energy_forecaster",
        "streamlit_url": "",
        "pages": [
            "Solar Dashboard",
            "Facility Analysis",
            "Production Forecast",
            "Model Performance",
            "About",
        ],
        "methodology": [
            "Collected monthly PV output from city solar installations",
            "Seasonal decomposition for production pattern analysis",
            "Regression models for production forecasting",
            "Site-level benchmarking and ROI calculations",
        ],
    },
    {
        "number": 9,
        "title": "Business Survival Recommender",
        "tagline": "Analyze business longevity and recommend locations",
        "description": (
            "Applies survival analysis to 22K+ business licenses to identify "
            "factors driving business longevity. Recommends optimal community "
            "locations based on business type, demographics, and competitive "
            "landscape."
        ),
        "categories": ["Survival Analysis", "Recommendation"],
        "tech_stack": ["Kaplan-Meier", "Cox PH", "lifelines", "Plotly", "Streamlit"],
        "dataset": "Business Licenses",
        "dataset_size": "22K+ records",
        "key_metric": "Concordance index for survival model",
        "folder": "project_09_business_survival_recommender",
        "streamlit_url": "",
        "pages": [
            "Business Dashboard",
            "Survival Analysis",
            "Location Recommender",
            "Model Performance",
            "About",
        ],
        "methodology": [
            "Computed business lifetimes from licence issue/cancel dates",
            "Kaplan-Meier survival curves by business category",
            "Cox Proportional Hazards model for risk factors",
            "Content-based location recommendation engine",
        ],
    },
    {
        "number": 10,
        "title": "Water Quality Anomaly Detection",
        "tagline": "Detect anomalies in watershed water quality data",
        "description": (
            "Monitors multi-parameter water quality across Calgary's watershed "
            "using anomaly detection. Features real-time alerts, parameter "
            "correlation analysis, and historical anomaly tracking for "
            "environmental protection."
        ),
        "categories": ["Anomaly Detection"],
        "tech_stack": ["Isolation Forest", "LOF", "Statistical Methods", "Plotly", "Streamlit"],
        "dataset": "Water Quality Data",
        "dataset_size": "Multi-site records",
        "key_metric": "Anomaly detection precision/recall",
        "folder": "project_10_water_quality_anomaly_detection",
        "streamlit_url": "",
        "pages": [
            "Water Quality Dashboard",
            "Anomaly Detection",
            "Site Analysis",
            "Parameter Correlations",
            "About",
        ],
        "methodology": [
            "Multi-parameter water quality data across monitoring sites",
            "Statistical process control for threshold-based alerts",
            "Isolation Forest and Local Outlier Factor for anomaly detection",
            "Cross-parameter correlation and trend analysis",
        ],
    },
    {
        "number": 11,
        "title": "311 Service Request Router",
        "tagline": "Auto-route citizen requests to the right city department",
        "description": (
            "Applies NLP text classification to Calgary's 311 service requests "
            "to automatically predict department routing and estimate resolution "
            "times. Features multi-label classification on request descriptions, "
            "category trend analysis, and response-time benchmarking across "
            "city departments."
        ),
        "categories": ["NLP", "Classification"],
        "tech_stack": ["TF-IDF", "BERT Embeddings", "Multi-label Classification", "scikit-learn", "Plotly", "Streamlit"],
        "dataset": "311 Service Requests",
        "dataset_size": "500K+ records",
        "key_metric": "Multi-label routing accuracy",
        "folder": "project_11_311_service_request_router",
        "streamlit_url": "",
        "pages": [
            "Request Dashboard",
            "Auto-Router",
            "Department Analysis",
            "Model Performance",
            "About",
        ],
        "methodology": [
            "Parsed 311 request text with TF-IDF and BERT embeddings",
            "Multi-label classification for department routing",
            "Resolution time estimation with regression models",
            "Department workload and trend analysis",
        ],
    },
    {
        "number": 12,
        "title": "Property Assessment Valuator",
        "tagline": "Estimate property values with explainable ML",
        "description": (
            "Predicts property assessment values across Calgary using geospatial "
            "features, neighborhood characteristics, and property attributes. "
            "Features SHAP-based model explainability showing exactly which "
            "factors drive each valuation, community comparison tools, and "
            "historical assessment trend analysis."
        ),
        "categories": ["Regression", "Explainability"],
        "tech_stack": ["XGBoost", "SHAP", "Geospatial Features", "Plotly", "Streamlit"],
        "dataset": "Property Assessments",
        "dataset_size": "500K+ properties",
        "key_metric": "Valuation MAPE with SHAP explanations",
        "folder": "project_12_property_assessment_valuator",
        "streamlit_url": "",
        "pages": [
            "Assessment Dashboard",
            "Property Valuator",
            "SHAP Explainer",
            "Model Performance",
            "About",
        ],
        "methodology": [
            "Merged property attributes with geospatial neighborhood features",
            "XGBoost regression with hyperparameter tuning",
            "SHAP values for per-prediction explainability",
            "Community-level valuation comparison and trend analysis",
        ],
    },
    {
        "number": 13,
        "title": "Transit Ridership Optimizer",
        "tagline": "Forecast ridership and optimize transit schedules",
        "description": (
            "Analyzes Calgary Transit ridership data across routes and stops "
            "using graph network analysis and demand forecasting. Models the "
            "transit network as a graph to identify bottlenecks, predict "
            "ridership by route and time of day, and simulate schedule "
            "optimization scenarios for improved service coverage."
        ),
        "categories": ["Time Series", "Network Analysis"],
        "tech_stack": ["NetworkX", "Prophet", "Graph Analysis", "Optimization", "Plotly", "Streamlit"],
        "dataset": "Transit Ridership",
        "dataset_size": "Route & stop records",
        "key_metric": "Ridership forecast accuracy & network efficiency",
        "folder": "project_13_transit_ridership_optimizer",
        "streamlit_url": "",
        "pages": [
            "Transit Dashboard",
            "Network Graph",
            "Ridership Forecast",
            "Route Optimizer",
            "About",
        ],
        "methodology": [
            "Modeled transit network as graph with NetworkX",
            "Centrality and bottleneck analysis on route network",
            "Prophet-based ridership demand forecasting",
            "Schedule optimization simulation for coverage improvement",
        ],
    },
    {
        "number": 27,
        "title": "Wildfire Risk Forecaster",
        "tagline": "Predict wildfire probability from remote sensing and weather data",
        "description": (
            "Combines satellite imagery, vegetation indices (NDVI/EVI), and weather "
            "data from ODAA to build spatial risk models predicting wildfire "
            "probability across Alberta's six monitored regions. Features Prophet "
            "time-series forecasting and SHAP explainability."
        ),
        "categories": ["Time Series", "Classification", "Spatial Analysis"],
        "tech_stack": ["XGBoost", "Prophet", "GeoPandas", "Rasterio", "SHAP", "Streamlit"],
        "dataset": "ODAA Remote Sensing + Weather",
        "dataset_size": "6 regions, GeoTIFF + CSV",
        "key_metric": "AUC-ROC with XGBoost",
        "folder": "project_27_wildfire_risk_forecaster",
        "streamlit_url": "",
        "pages": ["Risk Heatmap", "Vegetation Analysis", "Forecast Engine", "Model Explainability"],
        "methodology": [
            "Ingest GeoTIFF and shapefiles from ODAA",
            "Compute NDVI/EVI vegetation indices",
            "Merge weather and fire history data",
            "Train XGBoost spatial risk classifier",
            "Forecast seasonal risk with Prophet",
        ],
    },
    {
        "number": 28,
        "title": "Environmental Anomaly Detector",
        "tagline": "Unsupervised detection of pollution and deforestation events",
        "description": (
            "Builds an ensemble anomaly detection system on ODAA environmental "
            "sensor data using Isolation Forest, LOF, and LSTM Autoencoders. "
            "Flags pollution spikes, deforestation, and abnormal land surface "
            "temperature patterns across 109 datasets in six regions."
        ),
        "categories": ["Anomaly Detection"],
        "tech_stack": ["Isolation Forest", "LSTM Autoencoder", "LOF", "GeoPandas", "Rasterio", "Streamlit"],
        "dataset": "ODAA Environmental Sensors",
        "dataset_size": "109 datasets, 6 regions",
        "key_metric": "Precision at 5% contamination rate",
        "folder": "project_28_environmental_anomaly_detector",
        "streamlit_url": "",
        "pages": ["Anomaly Timeline", "Spatial Alerts", "Sensor Deep-dive", "Model Comparison"],
        "methodology": [
            "Ingest raster and vector data from ODAA",
            "Extract environmental time series per cell",
            "Apply SPC charts as first-pass filter",
            "Train Isolation Forest and LOF models",
            "Build LSTM Autoencoder for sequences",
            "Ensemble vote and deploy dashboard",
        ],
    },
    {
        "number": 29,
        "title": "Land Use Segmentation & Clustering",
        "tagline": "Unsupervised clustering of Alberta land into use profiles",
        "description": (
            "Applies K-Means, DBSCAN, and hierarchical clustering to geospatial "
            "and environmental features from six ODAA regions, segmenting the "
            "landscape into seven distinct land-use profiles. Includes PCA "
            "dimensionality reduction and silhouette analysis."
        ),
        "categories": ["Clustering", "Dimensionality Reduction", "Spatial Analysis"],
        "tech_stack": ["K-Means", "DBSCAN", "PCA", "GeoPandas", "Plotly", "Streamlit"],
        "dataset": "ODAA Shapefiles + Rasters",
        "dataset_size": "48 shapefiles, 6 regions",
        "key_metric": "Silhouette score (K-Means, k=7)",
        "folder": "project_29_land_use_segmentation",
        "streamlit_url": "",
        "pages": ["Cluster Map", "Feature Explorer", "Dimensionality Reduction", "Cluster Diagnostics"],
        "methodology": [
            "Load shapefiles and rasters from ODAA",
            "Extract terrain and vegetation features",
            "Standardize and reduce with PCA",
            "Compare K-Means, DBSCAN, hierarchical",
            "Select optimal k via silhouette analysis",
        ],
    },
    {
        "number": 30,
        "title": "Satellite Image Classifier",
        "tagline": "CNN land cover classification from Maxar satellite imagery",
        "description": (
            "Fine-tunes a ResNet-50 CNN on Maxar/DigitalGlobe GeoTIFF tiles "
            "from ODAA to classify image patches into six land cover categories. "
            "Includes transfer learning, data augmentation, and Grad-CAM "
            "attention maps for model interpretability."
        ),
        "categories": ["Classification"],
        "tech_stack": ["PyTorch", "ResNet-50", "Rasterio", "Grad-CAM", "Plotly", "Streamlit"],
        "dataset": "ODAA Maxar GeoTIFF",
        "dataset_size": "Satellite tiles, 6 regions",
        "key_metric": "Top-1 accuracy (ResNet-50)",
        "folder": "project_30_satellite_image_classifier",
        "streamlit_url": "",
        "pages": ["Image Explorer", "Classification Map", "Confusion Matrix", "Grad-CAM Attention"],
        "methodology": [
            "Extract tiles from Maxar GeoTIFFs",
            "Band selection and normalization",
            "Data augmentation (rotations, flips)",
            "Fine-tune ResNet-50 with transfer learning",
            "Evaluate with confusion matrix and F1",
            "Generate Grad-CAM attention maps",
        ],
    },
    {
        "number": 31,
        "title": "Economic Impact Dashboard",
        "tagline": "Quantify how environmental changes affect regional economies",
        "description": (
            "Merges ODAA environmental datasets with Alberta socioeconomic "
            "indicators to model how land degradation, wildfire, and vegetation "
            "loss correlate with GDP, property values, and employment. Features "
            "what-if simulation and auto-generated executive summaries."
        ),
        "categories": ["Regression", "Spatial Analysis"],
        "tech_stack": ["Ridge Regression", "Panel Models", "GeoPandas", "Plotly", "Streamlit"],
        "dataset": "ODAA + Alberta Economic Data",
        "dataset_size": "Environmental + economic, 6 regions",
        "key_metric": "R-squared (economic impact model)",
        "folder": "project_31_economic_impact_dashboard",
        "streamlit_url": "",
        "pages": ["Regional Overview", "Correlation Explorer", "Impact Simulator", "Executive Summary"],
        "methodology": [
            "Merge environmental and economic datasets",
            "Build regional panel regression models",
            "Quantify environment-to-economy linkages",
            "Create what-if impact simulator",
            "Auto-generate executive narratives",
        ],
    },
    {
        "number": 32,
        "title": "Site Suitability Recommender",
        "tagline": "Multi-criteria recommender for optimal land-use allocation",
        "description": (
            "Builds a multi-criteria decision analysis (MCDA) recommender on "
            "ODAA geospatial data, scoring grid cells for development, "
            "agriculture, or conservation suitability based on user-defined "
            "weights for terrain, water, soil, fire risk, and infrastructure."
        ),
        "categories": ["Recommendation", "Spatial Analysis"],
        "tech_stack": ["MCDA / AHP", "GeoPandas", "scikit-learn", "Folium", "Plotly", "Streamlit"],
        "dataset": "ODAA Multi-layer Geospatial",
        "dataset_size": "8 criteria layers, 6 regions",
        "key_metric": "NDCG@10 (recommendation quality)",
        "folder": "project_32_site_suitability_recommender",
        "streamlit_url": "",
        "pages": ["Criteria Builder", "Suitability Map", "Top-k Sites", "Scenario Comparison"],
        "methodology": [
            "Assemble multi-layer geospatial features",
            "Normalize criteria to 0-1 scale",
            "Implement AHP pairwise weighting",
            "Score cells via weighted linear combination",
            "Validate with NDCG ranking metrics",
        ],
    },
]

# ── Skills matrix: technique → list of project numbers that use it ──────────
SKILLS_MATRIX = {
    "Linear/Ridge Regression": [1],
    "Random Forest": [1],
    "XGBoost / Gradient Boosting": [1, 2, 5, 12],
    "Logistic Regression": [2, 7],
    "Decision Trees": [2],
    "DBSCAN": [3],
    "K-Means": [3, 6],
    "ARIMA / SARIMA": [4],
    "Prophet": [4, 5, 13],
    "LSTM (Deep Learning)": [4],
    "PCA": [6],
    "Hierarchical Clustering": [6],
    "TF-IDF / NLP": [7, 11],
    "Seasonal Decomposition": [8],
    "Kaplan-Meier / Cox PH": [9],
    "Isolation Forest": [10],
    "Local Outlier Factor": [10],
    "Statistical Process Control": [10],
    "BERT Embeddings": [11],
    "Multi-label Classification": [11],
    "SHAP Explainability": [12],
    "Graph / Network Analysis": [13],
    "Route Optimization": [13],
}

# ── Tools & technologies grid ───────────────────────────────────────────────
TOOLS = {
    "Languages": ["Python 3.10+"],
    "Data Processing": ["pandas", "NumPy", "SciPy"],
    "Machine Learning": ["scikit-learn", "XGBoost", "LightGBM", "TensorFlow", "SHAP", "NetworkX"],
    "Visualization": ["Plotly", "Seaborn", "Matplotlib"],
    "Web Applications": ["Streamlit"],
    "Data Access": ["Socrata API (SODA)", "sodapy"],
    "Version Control": ["Git", "GitHub"],
}

# ── Domain expertise ────────────────────────────────────────────────────────
DOMAINS = [
    {"name": "Urban Transportation", "projects": [3, 13], "icon": "🚗"},
    {"name": "Public Safety", "projects": [2, 11], "icon": "🛡️"},
    {"name": "Environmental Monitoring", "projects": [4, 10], "icon": "🌊"},
    {"name": "Construction & Real Estate", "projects": [1, 7, 12], "icon": "🏗️"},
    {"name": "Social Services", "projects": [5], "icon": "🏠"},
    {"name": "Renewable Energy", "projects": [8], "icon": "☀️"},
    {"name": "Business Analytics", "projects": [9], "icon": "💼"},
    {"name": "Urban Planning", "projects": [6], "icon": "🏙️"},
]
