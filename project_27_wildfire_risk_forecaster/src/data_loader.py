"""
Data loader for Wildfire Risk Forecaster.
Ingests GeoTIFF, shapefiles, and CSV data from ODAA sources.
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
    """Read a GeoTIFF file and return array + metadata."""
    with rasterio.open(filepath) as src:
        data = src.read()
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
    return data, meta, transform, crs


def compute_ndvi(red_band, nir_band):
    """Compute Normalized Difference Vegetation Index."""
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    ndvi = np.where(
        (nir + red) == 0, 0, (nir - red) / (nir + red)
    )
    return np.clip(ndvi, -1, 1)


def compute_evi(red_band, nir_band, blue_band, G=2.5, C1=6.0, C2=7.5, L=1.0):
    """Compute Enhanced Vegetation Index."""
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    blue = blue_band.astype(np.float32)
    denom = nir + C1 * red - C2 * blue + L
    evi = np.where(denom == 0, 0, G * (nir - red) / denom)
    return np.clip(evi, -1, 1)


def load_shapefiles(region_name):
    """Load region boundary shapefiles."""
    shp_dir = DATA_DIR / "shapefiles" / region_name
    if not shp_dir.exists():
        return None
    shp_files = list(shp_dir.glob("*.shp"))
    if not shp_files:
        return None
    return gpd.read_file(shp_files[0])


def load_weather_data():
    """Load merged weather station data."""
    filepath = DATA_DIR / "weather_stations.csv"
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


def load_fire_history():
    """Load historical fire occurrence records."""
    filepath = DATA_DIR / "fire_history.csv"
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_csv(filepath, parse_dates=["fire_date"])
    return df


def build_feature_matrix(regions=None):
    """
    Build the full feature matrix for model training.
    Combines vegetation indices, weather, and terrain features
    per spatial grid cell.
    """
    config = load_config()
    if regions is None:
        regions = config["data"]["regions"]

    frames = []
    for region in regions:
        gdf = load_shapefiles(region)
        if gdf is None:
            continue

        weather = load_weather_data()
        if not weather.empty:
            region_weather = weather[weather["region"] == region]
        else:
            region_weather = pd.DataFrame()

        record = {
            "region": region,
            "geometry_count": len(gdf) if gdf is not None else 0,
            "weather_records": len(region_weather),
        }
        frames.append(record)

    return pd.DataFrame(frames)


def get_engineered_data():
    """Load pre-engineered feature matrix if available."""
    filepath = DATA_DIR / "engineered_data.csv"
    if filepath.exists():
        return pd.read_csv(filepath)
    return build_feature_matrix()
