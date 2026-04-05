"""
Step 03 — Assign Administrative Regions to Each Grid Point

Each RWI grid cell is assigned to:
  1. A GADM Level-1 parish (GID_1, NAME_1)
  2. A poverty-data subregion: "Urban", "Rural", or "Kingston Metropolitan Area (KMA)"

The subregion assignment logic is:
  - KMA  = points in Kingston (JAM.3_1) OR Saint Andrew (JAM.6_1) parishes
            that are classified as urban (SMOD is_urban == 1)
  - Urban = all other urban points (is_urban == 1 and not KMA)
  - Rural = all non-urban points (is_urban == 0)

This mapping is the spatial bridge between fine-grained proxy data and the
coarse poverty statistics in ChPov_JAM_CUB.xlsx.

Assumptions logged:
  - Points that fall outside all GADM parish polygons (e.g. coastal pixels)
    receive parish_name="Unknown" and are excluded from reconciliation.
  - KMA classification uses GADM GID codes JAM.3_1 and JAM.6_1, as defined
    in config.yaml.  Any change in GADM version may require updating these codes.

Output saved to:  cfg["paths"]["grid_with_admin_file"]   (Parquet)

Added columns:
  gid_1        : GADM Level-1 GID string (e.g. "JAM.3_1")
  parish_name  : GADM Level-1 parish name
  subregion    : one of {"Urban", "Rural", "Kingston Metropolitan Area (KMA)"}
"""

import logging
import os

import geopandas as gpd
import pandas as pd

from src.utils.config_loader import load_config, setup_logging
from src.utils.geo_utils import (
    points_to_geodataframe,
    spatial_join_points_to_polygons,
    ensure_crs,
)

logger = logging.getLogger(__name__)


def _load_gadm_parishes(cfg: dict) -> gpd.GeoDataFrame:
    """Load the GADM Level-1 parish boundaries."""
    gadm_path = cfg["paths"]["gadm_gpkg"]
    layer = cfg["geo"]["gadm_layer"]
    logger.info("Loading GADM boundaries from: %s  (layer: %s)", gadm_path, layer)

    if not os.path.isfile(gadm_path):
        raise FileNotFoundError(f"GADM GeoPackage not found: {gadm_path}")

    gdf = gpd.read_file(gadm_path, layer=layer)
    logger.info("Loaded %d parishes. CRS: %s", len(gdf), gdf.crs)
    logger.info("Parish names: %s", sorted(gdf["NAME_1"].tolist()))

    target_crs = cfg["geo"]["crs"]
    gdf = ensure_crs(gdf, target_crs)
    return gdf


def _assign_subregion(row: pd.Series, kma_gids: list) -> str:
    """
    Derive the poverty-data subregion label for a single grid row.

    Parameters
    ----------
    row : pd.Series
        Must contain: gid_1, is_urban
    kma_gids : list of str
        GID codes for KMA parishes (Kingston, Saint Andrew).

    Returns
    -------
    str : "Urban", "Rural", or "Kingston Metropolitan Area (KMA)"
    """
    if pd.isna(row.get("gid_1")):
        return "Unknown"

    in_kma_parish = row["gid_1"] in kma_gids
    is_urban = bool(row.get("is_urban", 0))

    if in_kma_parish and is_urban:
        return "Kingston Metropolitan Area (KMA)"
    elif is_urban:
        return "Urban"
    else:
        return "Rural"


def assign_admin(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Spatially assign parish and subregion labels to the grid.

    Parameters
    ----------
    cfg : dict
        Loaded config.
    df : pd.DataFrame
        Grid with proxy features (from Step 02).

    Returns
    -------
    pd.DataFrame
        Grid with added admin columns.
    """
    target_crs = cfg["geo"]["crs"]
    kma_gids = cfg["geo"]["kma_parish_gids"]

    # ------------------------------------------------------------------
    # Load GADM parishes
    # ------------------------------------------------------------------
    parishes = _load_gadm_parishes(cfg)

    # ------------------------------------------------------------------
    # Convert grid to GeoDataFrame
    # ------------------------------------------------------------------
    logger.info("Converting %d grid points to GeoDataFrame...", len(df))
    points_gdf = points_to_geodataframe(df, crs=target_crs)

    # ------------------------------------------------------------------
    # Spatial join — points within parishes
    # ------------------------------------------------------------------
    logger.info("Running spatial join (points within parishes)...")
    joined = spatial_join_points_to_polygons(
        points_gdf,
        parishes,
        polygon_cols=["GID_1", "NAME_1"],
    )

    n_unmatched = joined["GID_1"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "%d grid points did not fall within any parish polygon. "
            "These will be assigned subregion='Unknown'.",
            n_unmatched,
        )

    # Rename parish columns to lowercase
    joined = joined.rename(columns={"GID_1": "gid_1", "NAME_1": "parish_name"})

    # ------------------------------------------------------------------
    # Assign poverty-data subregion
    # ------------------------------------------------------------------
    logger.info("Assigning subregion labels (Urban / Rural / KMA)...")
    joined["subregion"] = joined.apply(
        lambda row: _assign_subregion(row, kma_gids), axis=1
    )

    # Log assumption about KMA definition
    logger.info(
        "ASSUMPTION: KMA defined as urban points in parishes %s. "
        "This corresponds to the 'Kingston Metropolitan Area (KMA)' subregion "
        "in ChPov_JAM_CUB.xlsx. Verify against official KMA boundary if needed.",
        kma_gids,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    subregion_counts = joined["subregion"].value_counts()
    logger.info("Subregion assignment summary:\n%s", subregion_counts.to_string())

    parish_counts = joined["parish_name"].value_counts()
    logger.info("Parish assignment summary:\n%s", parish_counts.to_string())

    # Convert back to plain DataFrame (drop geometry column)
    result = pd.DataFrame(joined.drop(columns=["geometry"]))
    return result


def run(cfg: dict, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Entry point for Step 03.

    Parameters
    ----------
    cfg : dict
    df : pd.DataFrame or None
        Grid with proxies.  If None, loads from cfg["paths"]["grid_with_proxies_file"].

    Returns
    -------
    pd.DataFrame
    """
    if df is None:
        src_path = cfg["paths"]["grid_with_proxies_file"]
        logger.info("Loading grid with proxies from: %s", src_path)
        df = pd.read_parquet(src_path)

    result = assign_admin(cfg, df)

    out_path = cfg["paths"]["grid_with_admin_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_parquet(out_path, index=False)
    logger.info("Grid with admin assignment saved to: %s  (%d rows)", out_path, len(result))

    return result


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    run(_cfg)
