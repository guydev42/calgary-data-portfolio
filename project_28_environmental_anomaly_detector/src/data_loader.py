"""
Data loader for Environmental Anomaly Detector.
Ingests environmental raster and vector data from ODAA sources.
"""

import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def load_config():
    with open(PROJECT_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def load_raster(filepath):
    """Read a GeoTIFF and return array + metadata."""
    with rasterio.open(filepath) as src:
        return src.read(), src.meta.copy(), src.transform, src.crs


def extract_land_surface_temperature(thermal_band):
    """Convert thermal band DN to land surface temperature (Celsius)."""
    radiance = thermal_band.astype(np.float32) * 0.0003342 + 0.1
    bt = 1321.08 / np.log(774.89 / radiance + 1) - 273.15
    return bt


def load_environmental_timeseries():
    """Load pre-processed environmental time series."""
    filepath = DATA_DIR / "env_timeseries.csv"
    if not filepath.exists():
        return pd.DataFrame()
    return pd.read_csv(filepath, parse_dates=["timestamp"])


def load_shapefiles(region_name):
    """Load region boundary shapefiles."""
    shp_dir = DATA_DIR / "shapefiles" / region_name
    if not shp_dir.exists():
        return None
    shp_files = list(shp_dir.glob("*.shp"))
    if not shp_files:
        return None
    return gpd.read_file(shp_files[0])


def build_feature_matrix():
    """Build multivariate feature matrix from environmental sensors."""
    config = load_config()
    df = load_environmental_timeseries()
    if df.empty:
        return pd.DataFrame()

    features = [
        "land_surface_temp", "spectral_reflectance", "ndvi",
        "soil_moisture", "air_quality_index", "precipitation",
    ]
    available = [f for f in features if f in df.columns]
    return df[["timestamp", "region", "grid_cell"] + available]


def get_engineered_data():
    """Load pre-engineered data if available."""
    filepath = DATA_DIR / "engineered_data.csv"
    if filepath.exists():
        return pd.read_csv(filepath)
    return build_feature_matrix()
