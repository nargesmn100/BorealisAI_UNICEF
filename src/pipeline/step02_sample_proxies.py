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
from scipy.spatial import cKDTree

from src.utils.config_loader import load_config, setup_logging
from src.utils.geo_utils import (
    open_raster_from_zip,
    sample_raster_at_points,
    decode_smod,
    SMOD_URBAN_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _impute_population_spatial(
    lons: np.ndarray,
    lats: np.ndarray,
    pop_values: np.ndarray,
    method: str = "knn",
    n_neighbors: int = 8,
    max_distance_deg: float = 0.1,
) -> tuple:
    """
    Spatially impute missing population values using KNN or IDW.

    Parameters
    ----------
    lons, lats : np.ndarray
        Coordinates of all grid cells.
    pop_values : np.ndarray
        Population values (NaN for missing).
    method : str
        "knn" for simple mean, "idw" for inverse-distance weighted mean.
    n_neighbors : int
        Number of nearest neighbors to use.
    max_distance_deg : float
        Maximum search distance in degrees.

    Returns
    -------
    (imputed_values, imputed_flag) : (np.ndarray, np.ndarray)
        imputed_values: population with NaN filled where possible.
        imputed_flag: boolean array (True for imputed cells).
    """
    valid_mask = ~np.isnan(pop_values) & (pop_values >= 0)
    missing_mask = np.isnan(pop_values)

    n_missing = missing_mask.sum()
    if n_missing == 0:
        return pop_values.copy(), np.zeros(len(pop_values), dtype=bool)

    logger.info(
        "Spatially imputing %d missing population values using %s (k=%d, max_dist=%.3f)...",
        n_missing, method, n_neighbors, max_distance_deg,
    )

    # Build KDTree from valid cells
    valid_coords = np.column_stack([lons[valid_mask], lats[valid_mask]])
    valid_pop = pop_values[valid_mask]
    tree = cKDTree(valid_coords)

    # Query for each missing cell
    missing_coords = np.column_stack([lons[missing_mask], lats[missing_mask]])
    distances, indices = tree.query(missing_coords, k=n_neighbors)

    # Handle case where k=1 returns 1D arrays
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    result = pop_values.copy()
    imputed_flag = np.zeros(len(pop_values), dtype=bool)

    missing_indices = np.where(missing_mask)[0]
    n_imputed = 0

    for i, cell_idx in enumerate(missing_indices):
        # Filter neighbors within max_distance
        within_range = distances[i] <= max_distance_deg
        if not within_range.any():
            continue

        neighbor_dists = distances[i][within_range]
        neighbor_pops = valid_pop[indices[i][within_range]]

        if method == "idw":
            # Inverse-distance weighting (add small epsilon to avoid div by zero)
            weights = 1.0 / (neighbor_dists + 1e-10)
            result[cell_idx] = np.average(neighbor_pops, weights=weights)
        else:
            # Simple KNN mean
            result[cell_idx] = np.mean(neighbor_pops)

        imputed_flag[cell_idx] = True
        n_imputed += 1

    logger.info(
        "Population imputation: %d of %d missing cells filled. %d remain (beyond max_distance).",
        n_imputed, n_missing, n_missing - n_imputed,
    )

    return result, imputed_flag


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
    # 1. Population (with optional spatial imputation)
    # ------------------------------------------------------------------
    pop = _sample_population(cfg, lons, lats)

    pop_impute_cfg = cfg["geo"].get("population_imputation", {})
    if pop_impute_cfg.get("enabled", False):
        pop, pop_imputed_flag = _impute_population_spatial(
            lons, lats, pop,
            method=pop_impute_cfg.get("method", "knn"),
            n_neighbors=pop_impute_cfg.get("n_neighbors", 8),
            max_distance_deg=pop_impute_cfg.get("max_distance_deg", 0.1),
        )
        df["population_imputed"] = pop_imputed_flag.astype(int)
    else:
        df["population_imputed"] = 0

    df["population"] = pop
    df["log_population"] = np.log1p(np.where(np.isnan(pop), 0.0, pop))

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
    # 4. Nighttime lights (optional — per problem statement §3/§5)
    # ------------------------------------------------------------------
    nl_path = cfg["paths"].get("nightlights_raster")
    if nl_path and os.path.isfile(nl_path):
        logger.info("Sampling nighttime lights raster from: %s", nl_path)
        import rasterio
        with rasterio.open(nl_path) as src:
            nl_values = sample_raster_at_points(
                src, lons, lats, band=1, fill_value=np.nan
            )
        # Clamp negatives to 0
        nl_values = np.where((nl_values < 0) | np.isnan(nl_values), 0.0, nl_values)
        df["nightlights"] = nl_values
        df["log_nightlights"] = np.log1p(nl_values)
        logger.info(
            "Nightlights — min: %.2f, max: %.2f, mean: %.2f, missing: %d",
            float(np.nanmin(nl_values)), float(np.nanmax(nl_values)),
            float(np.nanmean(nl_values)), int(np.isnan(nl_values).sum()),
        )
    else:
        logger.info("Nightlights raster not available. Skipping.")

    # ------------------------------------------------------------------
    # 5. Building density (optional — per problem statement §3/§5)
    # ------------------------------------------------------------------
    bd_path = cfg["paths"].get("building_density_raster")
    if bd_path and os.path.isfile(bd_path):
        logger.info("Sampling building density raster from: %s", bd_path)
        import rasterio
        with rasterio.open(bd_path) as src:
            bd_values = sample_raster_at_points(
                src, lons, lats, band=1, fill_value=np.nan
            )
        bd_values = np.where((bd_values < 0) | np.isnan(bd_values), 0.0, bd_values)
        df["building_density"] = bd_values
        logger.info(
            "Building density — min: %.2f, max: %.2f, mean: %.2f, missing: %d",
            float(np.nanmin(bd_values)), float(np.nanmax(bd_values)),
            float(np.nanmean(bd_values)), int(np.isnan(bd_values).sum()),
        )
    else:
        logger.info("Building density raster not available. Skipping.")

    # ------------------------------------------------------------------
    # Data quality summary
    # ------------------------------------------------------------------
    logger.info("Proxy sampling complete. Summary of missing values:")
    proxy_cols = [
        "population", "smod_class", "travel_time_cities", "travel_time_50k"
    ]
    # Include optional features if present
    for optional in ["nightlights", "building_density"]:
        if optional in df.columns:
            proxy_cols.append(optional)
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
