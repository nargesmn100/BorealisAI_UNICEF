"""
Step 03 — Assign Administrative Regions to Each Grid Point

Each RWI grid cell is assigned to:
  1. A GADM Level-1 admin unit (GID_1, NAME_1)
  2. A poverty-data subregion label

Subregion assignment is config-driven via cfg["geo"]["subregion_strategy"]:
  - "kma_urban_rural" (Jamaica): Urban / Rural / Kingston Metropolitan Area (KMA)
  - "admin_level1" (Nigeria etc.): subregion = GADM NAME_1 directly (state name)

Output saved to:  cfg["paths"]["grid_with_admin_file"]   (Parquet)

Added columns:
  gid_1        : GADM Level-1 GID string
  parish_name  : GADM Level-1 admin unit name
  subregion    : poverty-data subregion label
"""

import logging
import os

import geopandas as gpd
import pandas as pd

from src.utils.config_loader import load_config, setup_logging
from src.utils.geo_utils import (
    points_to_geodataframe,
    spatial_join_points_to_polygons,
    spatial_join_with_nearest_fallback,
    ensure_crs,
)

logger = logging.getLogger(__name__)


def _load_gadm_parishes(cfg: dict) -> gpd.GeoDataFrame:
    """Load the GADM Level-1 parish boundaries with defensive layer detection."""
    gadm_path = cfg["paths"]["gadm_gpkg"]
    layer = cfg["geo"]["gadm_layer"]
    logger.info("Loading GADM boundaries from: %s  (layer: %s)", gadm_path, layer)

    if not os.path.isfile(gadm_path):
        raise FileNotFoundError(f"GADM GeoPackage not found: {gadm_path}")

    # List available layers for diagnostics and fallback
    try:
        import fiona
        available_layers = fiona.listlayers(gadm_path)
        logger.info("Available GADM layers: %s", available_layers)
    except ImportError:
        available_layers = None
        logger.debug("fiona not available for layer listing, proceeding with configured layer.")

    try:
        gdf = gpd.read_file(gadm_path, layer=layer)
    except Exception as e:
        if available_layers is not None:
            # Try fallback: first layer containing "1" (i.e., ADM level 1)
            fallback = next((l for l in available_layers if "1" in l), None)
            if fallback and fallback != layer:
                logger.warning(
                    "Configured layer '%s' not found (%s). "
                    "Falling back to '%s'.",
                    layer, e, fallback,
                )
                gdf = gpd.read_file(gadm_path, layer=fallback)
            else:
                raise
        else:
            raise

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


_SUBREGION_HARMONISE: dict[str, str] = {
    # Nigeria: GADM uses "Federal Capital Territory"; MICS SPSS may give "Fct"
    # All comparisons are done after .strip().title() so keys must be Title Case.
    "Federal Capital Territory": "Fct",
}


def _assign_subregion_admin_level1(row: pd.Series) -> str:
    """
    Assign subregion = GADM NAME_1 directly (for admin_level1 strategy).

    Applies ``_SUBREGION_HARMONISE`` to reconcile GADM labels with MICS SPSS
    value-label variants (e.g. "Federal Capital Territory" → "Fct").
    """
    name = row.get("parish_name")
    if pd.isna(name) or name == "":
        return "Unknown"
    name = str(name).strip()
    return _SUBREGION_HARMONISE.get(name, name)


def assign_admin(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Spatially assign admin unit and subregion labels to the grid.

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
    strategy = cfg["geo"].get("subregion_strategy", "kma_urban_rural")

    # ------------------------------------------------------------------
    # Load GADM admin units
    # ------------------------------------------------------------------
    parishes = _load_gadm_parishes(cfg)

    # ------------------------------------------------------------------
    # Convert grid to GeoDataFrame
    # ------------------------------------------------------------------
    logger.info("Converting %d grid points to GeoDataFrame...", len(df))
    points_gdf = points_to_geodataframe(df, crs=target_crs)

    # ------------------------------------------------------------------
    # Spatial join — points within admin polygons (with optional nearest fallback)
    # ------------------------------------------------------------------
    use_nearest_fallback = cfg["geo"].get("nearest_parish_fallback", False)
    max_distance = cfg["geo"].get("nearest_parish_max_distance", 0.05)

    if use_nearest_fallback:
        logger.info("Running spatial join with nearest-admin fallback (max_distance=%.4f)...", max_distance)
        joined = spatial_join_with_nearest_fallback(
            points_gdf,
            parishes,
            polygon_cols=["GID_1", "NAME_1"],
            max_distance=max_distance,
        )
    else:
        logger.info("Running spatial join (points within admin polygons)...")
        joined = spatial_join_points_to_polygons(
            points_gdf,
            parishes,
            polygon_cols=["GID_1", "NAME_1"],
        )
        joined["parish_imputed"] = False

    n_unmatched = joined["GID_1"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "%d grid points did not fall within any admin polygon. "
            "These will be assigned subregion='Unknown'.",
            n_unmatched,
        )

    # Rename columns to lowercase
    joined = joined.rename(columns={"GID_1": "gid_1", "NAME_1": "parish_name"})

    # ------------------------------------------------------------------
    # Assign poverty-data subregion (strategy-driven)
    # ------------------------------------------------------------------
    if strategy == "admin_level1":
        logger.info("Assigning subregion labels using admin_level1 strategy (subregion = state/admin name)...")
        joined["subregion"] = joined.apply(_assign_subregion_admin_level1, axis=1)
    else:
        # Default: Jamaica KMA/Urban/Rural strategy
        kma_gids = cfg["geo"].get("kma_parish_gids", [])
        logger.info("Assigning subregion labels (Urban / Rural / KMA)...")
        joined["subregion"] = joined.apply(
            lambda row: _assign_subregion(row, kma_gids), axis=1
        )
        logger.info(
            "ASSUMPTION: KMA defined as urban points in parishes %s. "
            "Verify against official KMA boundary if needed.",
            kma_gids,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    subregion_counts = joined["subregion"].value_counts()
    logger.info("Subregion assignment summary:\n%s", subregion_counts.to_string())

    parish_counts = joined["parish_name"].value_counts()
    logger.info("Admin unit assignment summary:\n%s", parish_counts.to_string())

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
