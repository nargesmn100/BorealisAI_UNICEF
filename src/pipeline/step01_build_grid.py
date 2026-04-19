"""
Step 01 — Build Base Grid

Loads the Jamaica Relative Wealth Index (RWI) CSV and converts it into the
canonical base geospatial grid used by all subsequent pipeline steps.

The RWI CSV is a ~2.3 km regular grid of 1 745 point observations
covering Jamaica's land area.  It contains:
  - latitude / longitude (WGS84)
  - rwi       : RWI score (float, ~N(0,1) standardised)
  - error     : per-observation uncertainty (std dev of posterior)

Output saved to:  cfg["paths"]["grid_file"]   (Parquet)

Columns in output:
  cell_id   : unique integer identifier
  latitude  : float
  longitude : float
  rwi       : float
  rwi_error : float (renamed from 'error' to avoid name collision)
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def build_base_grid(cfg: dict) -> pd.DataFrame:
    """
    Load the RWI CSV and produce the canonical base grid DataFrame.

    Parameters
    ----------
    cfg : dict
        Loaded config (see config/config.yaml).

    Returns
    -------
    pd.DataFrame
        Base grid with columns: cell_id, latitude, longitude, rwi, rwi_error.
    """
    rwi_path = cfg["paths"]["rwi_csv"]
    logger.info("Loading RWI base grid from: %s", rwi_path)

    if not os.path.isfile(rwi_path):
        raise FileNotFoundError(f"RWI CSV not found at: {rwi_path}")

    grid = pd.read_csv(rwi_path)

    # ------------------------------------------------------------------
    # Validate expected columns
    # ------------------------------------------------------------------
    expected_cols = {"latitude", "longitude", "rwi", "error"}
    missing = expected_cols - set(grid.columns)
    if missing:
        raise ValueError(
            f"RWI CSV is missing expected columns: {missing}. "
            f"Found: {list(grid.columns)}"
        )

    # ------------------------------------------------------------------
    # Rename and clean
    # ------------------------------------------------------------------
    grid = grid.rename(columns={"error": "rwi_error"})
    grid = grid.dropna(subset=["latitude", "longitude", "rwi"])
    grid = grid.reset_index(drop=True)
    grid.insert(0, "cell_id", grid.index.astype(int))

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    n_cells = len(grid)
    logger.info("Base grid contains %d cells (after dropping NaN rows).", n_cells)

    lat_range = (grid["latitude"].min(), grid["latitude"].max())
    lon_range = (grid["longitude"].min(), grid["longitude"].max())
    logger.info("Latitude range:  %.4f  –  %.4f", *lat_range)
    logger.info("Longitude range: %.4f  –  %.4f", *lon_range)

    # Crude bounding-box check
    country_name = cfg.get("country", {}).get("name", "country")
    bbox = cfg["geo"]["bbox"]
    if lat_range[0] < bbox["south"] or lat_range[1] > bbox["north"]:
        logger.warning(
            "Latitude range (%.4f, %.4f) partially outside %s bbox "
            "(%.4f, %.4f). Verify input data.",
            lat_range[0], lat_range[1],
            country_name, bbox["south"], bbox["north"],
        )
    if lon_range[0] < bbox["west"] or lon_range[1] > bbox["east"]:
        logger.warning(
            "Longitude range (%.4f, %.4f) partially outside %s bbox "
            "(%.4f, %.4f). Verify input data.",
            lon_range[0], lon_range[1],
            country_name, bbox["west"], bbox["east"],
        )

    rwi_stats = grid["rwi"].describe()
    logger.info(
        "RWI stats — mean: %.3f  std: %.3f  min: %.3f  max: %.3f",
        rwi_stats["mean"], rwi_stats["std"], rwi_stats["min"], rwi_stats["max"],
    )

    missing_rwi = grid["rwi"].isna().sum()
    if missing_rwi > 0:
        logger.warning("%d cells have missing RWI values.", missing_rwi)

    return grid


def run(cfg: dict) -> pd.DataFrame:
    """
    Entry point for Step 01.  Builds the base grid and saves it to disk.

    Parameters
    ----------
    cfg : dict
        Loaded config.

    Returns
    -------
    pd.DataFrame
        The base grid DataFrame.
    """
    grid = build_base_grid(cfg)

    out_path = cfg["paths"]["grid_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.to_parquet(out_path, index=False)
    logger.info("Base grid saved to: %s  (%d rows)", out_path, len(grid))

    return grid


if __name__ == "__main__":
    from src.utils.config_loader import load_config, setup_logging
    _cfg = load_config()
    setup_logging(_cfg)
    run(_cfg)
