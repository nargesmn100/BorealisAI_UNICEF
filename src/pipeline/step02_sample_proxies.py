"""
Step 02 — Sample Proxy Rasters onto the Base Grid

For each RWI grid point, this step samples:
  1. Population      (jam_pop_2030_CN_100m_R2025A_v1.tif)
  2. SMOD class      (GHS_SMOD, reprojected from ESRI:54009 → EPSG:4326)
  3. Travel time to cities (cit_017, already in EPSG:4326)
  4. Travel time to 50k cities (acc_50k, already in EPSG:4326)

Design decisions
----------------
- All rasters are sampled using nearest-neighbour lookup at RWI point coords.
- SMOD is a global raster in Mollweide (ESRI:54009).  We reproject on-the-fly
  by transforming the query points into ESRI:54009, sampling, then returning
  values.  This avoids the cost of reprojecting the full global raster.
- Population, travel-time rasters are in EPSG:4326 — direct sampling.
- Log-transforms are added for population and travel-time to reduce skew.
- SMOD class is decoded into a boolean `is_urban` flag.

Output saved to:  cfg["paths"]["grid_with_proxies_file"]   (Parquet)

Added columns (on top of base grid columns):
  population          : raw child-population count
  log_population      : log1p(population)
  smod_class          : raw GHSL SMOD integer class
  smod_label          : human-readable SMOD label
  is_urban            : bool (smod_class >= 21)
  travel_time_cities  : travel time to nearest city (minutes)
  travel_time_50k     : travel time to nearest 50 000-pop city (minutes)
  log_travel_time_cities : log1p(travel_time_cities)
  log_travel_time_50k    : log1p(travel_time_50k)
"""

import logging
import os

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config, setup_logging
from src.utils.geo_utils import (
    open_raster_from_zip,
    sample_raster_at_points,
    decode_smod,
    SMOD_URBAN_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _sample_population(cfg: dict, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Sample population raster at grid points."""
    pop_path = cfg["paths"]["population_raster"]
    logger.info("Sampling population raster from: %s", pop_path)
    import rasterio
    with rasterio.open(pop_path) as src:
        values = sample_raster_at_points(
            src, lons, lats, band=1, fill_value=np.nan
        )
    n_missing = np.isnan(values).sum()
    n_negative = (values < 0).sum()
    logger.info(
        "Population — missing: %d, negative (ocean/nodata): %d, "
        "max: %.2f, mean: %.4f",
        n_missing, n_negative,
        float(np.nanmax(values)),
        float(np.nanmean(values[values >= 0])) if np.any(values >= 0) else np.nan,
    )
    # Clamp negatives to 0 (these are coastal edge pixels)
    values = np.where(values < 0, 0.0, values)
    return values


def _sample_smod(cfg: dict, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Sample SMOD raster (ESRI:54009) at grid points (EPSG:4326).

    The pyproj Transformer converts WGS84 → Mollweide before sampling.
    """
    zip_path = cfg["paths"]["smod_zip"]
    tif_name = cfg["zip_contents"]["smod_tif_name"]
    logger.info("Sampling SMOD from zip: %s / %s", zip_path, tif_name)

    dataset = open_raster_from_zip(zip_path, tif_name)
    try:
        # sample_raster_at_points handles CRS reprojection internally
        values = sample_raster_at_points(
            dataset, lons, lats, band=1, fill_value=-200
        )
    finally:
        dataset.close()

    unique_classes, counts = np.unique(values.astype(int), return_counts=True)
    logger.info(
        "SMOD class distribution: %s",
        {int(k): int(v) for k, v in zip(unique_classes, counts)},
    )
    return values.astype(int)


def _sample_accessibility(
    cfg: dict, lons: np.ndarray, lats: np.ndarray
) -> tuple:
    """
    Sample both accessibility rasters.

    Returns
    -------
    (travel_time_cities, travel_time_50k) : (np.ndarray, np.ndarray)
    """
    # Travel time to cities (cit_017)
    logger.info("Sampling accessibility-to-cities raster...")
    ds_cities = open_raster_from_zip(
        cfg["paths"]["accessibility_zip"],
        cfg["zip_contents"]["accessibility_tif_name"],
    )
    try:
        tt_cities = sample_raster_at_points(
            ds_cities, lons, lats, band=1, fill_value=np.nan
        )
    finally:
        ds_cities.close()

    # The accessibility raster uses multiple nodata sentinels.
    # Replace any large negative value (< -1000) with NaN.
    tt_cities = np.where(tt_cities < -1000, np.nan, tt_cities)

    logger.info(
        "Travel-time-to-cities — missing: %d, min: %.1f, max: %.1f, mean: %.1f",
        int(np.isnan(tt_cities).sum()),
        float(np.nanmin(tt_cities)),
        float(np.nanmax(tt_cities)),
        float(np.nanmean(tt_cities)),
    )

    # Travel time to 50k cities (acc_50k)
    logger.info("Sampling acc_50k raster...")
    ds_50k = open_raster_from_zip(
        cfg["paths"]["access_50k_zip"],
        cfg["zip_contents"]["access_50k_tif_name"],
    )
    try:
        tt_50k = sample_raster_at_points(
            ds_50k, lons, lats, band=1, fill_value=np.nan
        )
    finally:
        ds_50k.close()

    logger.info(
        "Travel-time-to-50k — missing: %d, min: %.1f, max: %.1f, mean: %.1f",
        int(np.isnan(tt_50k).sum()),
        float(np.nanmin(tt_50k)),
        float(np.nanmax(tt_50k)),
        float(np.nanmean(tt_50k)),
    )

    return tt_cities, tt_50k


def sample_proxies(cfg: dict, grid: pd.DataFrame) -> pd.DataFrame:
    """
    Attach all proxy raster values to the base grid.

    Parameters
    ----------
    cfg : dict
        Loaded config.
    grid : pd.DataFrame
        Base grid from Step 01.

    Returns
    -------
    pd.DataFrame
        Grid with additional proxy columns.
    """
    lons = grid["longitude"].values
    lats = grid["latitude"].values

    df = grid.copy()

    # ------------------------------------------------------------------
    # 1. Population
    # ------------------------------------------------------------------
    pop = _sample_population(cfg, lons, lats)
    df["population"] = pop
    df["log_population"] = np.log1p(df["population"])

    # ------------------------------------------------------------------
    # 2. SMOD Settlement Class
    # ------------------------------------------------------------------
    smod = _sample_smod(cfg, lons, lats)
    df["smod_class"] = smod
    df["smod_label"] = pd.Series(smod).apply(decode_smod).values
    urban_threshold = cfg["geo"].get("smod_urban_threshold", SMOD_URBAN_THRESHOLD)
    df["is_urban"] = (smod >= urban_threshold).astype(int)

    n_urban = df["is_urban"].sum()
    logger.info(
        "Urban cells: %d / %d (%.1f%%)",
        n_urban, len(df), 100 * n_urban / len(df),
    )

    # ------------------------------------------------------------------
    # 3. Accessibility
    # ------------------------------------------------------------------
    tt_cities, tt_50k = _sample_accessibility(cfg, lons, lats)
    df["travel_time_cities"] = tt_cities
    df["travel_time_50k"] = tt_50k
    # For log transforms, clamp negative/NaN to 0 first
    tt_cities_clean = np.where(np.isnan(tt_cities) | (tt_cities < 0), 0.0, tt_cities)
    tt_50k_clean = np.where(np.isnan(tt_50k) | (tt_50k < 0), 0.0, tt_50k)
    df["log_travel_time_cities"] = np.log1p(tt_cities_clean)
    df["log_travel_time_50k"] = np.log1p(tt_50k_clean)

    # ------------------------------------------------------------------
    # Data quality summary
    # ------------------------------------------------------------------
    logger.info("Proxy sampling complete. Summary of missing values:")
    proxy_cols = [
        "population", "smod_class", "travel_time_cities", "travel_time_50k"
    ]
    for col in proxy_cols:
        n_missing = df[col].isna().sum()
        logger.info("  %s: %d missing", col, n_missing)

    return df


def run(cfg: dict, grid: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Entry point for Step 02.

    Parameters
    ----------
    cfg : dict
        Loaded config.
    grid : pd.DataFrame or None
        Base grid.  If None, loads from cfg["paths"]["grid_file"].

    Returns
    -------
    pd.DataFrame
        Grid with proxy features.
    """
    if grid is None:
        grid_path = cfg["paths"]["grid_file"]
        logger.info("Loading base grid from: %s", grid_path)
        grid = pd.read_parquet(grid_path)

    df = sample_proxies(cfg, grid)

    out_path = cfg["paths"]["grid_with_proxies_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Grid with proxies saved to: %s  (%d rows)", out_path, len(df))

    return df


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    run(_cfg)
