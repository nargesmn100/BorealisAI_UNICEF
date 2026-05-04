"""
LGA-Level Aggregation — Nigeria

Spatially joins RWI grid cell predictions to GADM ADM2 (Local Government Area)
boundaries and produces population-weighted aggregate estimates.

Outputs
-------
  Data/outputs/nga/tables/nga_lga_predictions.csv
      — flat table (775 LGAs): predictions, Ridge theme sums, full per-feature
        β·z contributions (prefixed ridge_bdg__*), and raw feature means.
  Data/outputs/nga/maps/nga_lga_predictions.geojson
      — polygon GIS layer (themes + predictions only; raw values omitted for size).

Usage
-----
  Called automatically by main.py phase_outputs() when country == NGA.
  Can also be run standalone:
    python -m src.outputs.lga_aggregation
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Prediction columns to aggregate (population-weighted mean)
_PRED_COLS = [
    "ridge_moderate", "ridge_severe", "ridge_moderate_depth",
    "gbm_moderate", "gbm_severe", "gbm_moderate_depth",
    "gam_moderate", "gam_severe",
    "wsnn_moderate", "wsnn_severe",
    "ridge_moderate_lower", "ridge_moderate_upper",
    "gbm_moderate_lower", "gbm_moderate_upper",
    "rwi_moderate", "heuristic_moderate", "uniform_moderate",
]

# Per-dimension prediction columns (Kyriaki spec)
_DIMENSION_COLS = [
    "shelter_moderate", "sanitation_moderate", "water_moderate",
    "nutrition_moderate", "edu_5_14_moderate", "edu_15_17_moderate",
    "health_moderate", "health_36_59_moderate",
]


def aggregate_to_lga(
    cfg: dict,
    pred_table: pd.DataFrame,
    gadm_path: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Aggregate grid cell predictions to GADM ADM2 (LGA) level.

    Parameters
    ----------
    cfg : dict
        Pipeline config dictionary.
    pred_table : pd.DataFrame
        Grid cell predictions table (must contain latitude, longitude, population).
    gadm_path : str, optional
        Path to GADM GeoPackage.  Defaults to cfg["paths"]["gadm_file"].

    Returns
    -------
    tuple[str | None, str | None]
        Paths to the saved CSV and GeoJSON files (or None if failed).
    """
    if gadm_path is None:
        gadm_path = cfg["paths"].get("gadm_file", cfg["paths"].get("gadm_gpkg", ""))

    if not os.path.isfile(gadm_path):
        logger.warning("GADM file not found: %s. Skipping LGA aggregation.", gadm_path)
        return None, None

    tables_dir = cfg["paths"]["tables_dir"]
    maps_dir   = cfg["paths"]["maps_dir"]
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(maps_dir,   exist_ok=True)

    output_prefix = cfg.get("country", {}).get("output_prefix", "nga")
    csv_path     = os.path.join(tables_dir, f"{output_prefix}_lga_predictions.csv")
    geojson_path = os.path.join(maps_dir,   f"{output_prefix}_lga_predictions.geojson")

    # -----------------------------------------------------------------------
    # Load LGA boundaries (GADM ADM2)
    # -----------------------------------------------------------------------
    logger.info("Loading GADM ADM2 boundaries from: %s", gadm_path)
    lga = gpd.read_file(gadm_path, layer="ADM_ADM_2")
    lga = lga[["GID_2", "NAME_1", "NAME_2", "geometry"]].rename(
        columns={"NAME_1": "state", "NAME_2": "lga_name", "GID_2": "lga_id"}
    )
    logger.info("LGAs loaded: %d", len(lga))

    # -----------------------------------------------------------------------
    # Optionally merge per-dimension predictions from nga_dimension_predictions.csv
    # -----------------------------------------------------------------------
    dim_pred_path = os.path.join(
        cfg["paths"].get("tables_dir", "Data/outputs/nga/tables"),
        f"{output_prefix}_dimension_predictions.csv",
    )
    if os.path.isfile(dim_pred_path):
        try:
            dim_preds = pd.read_csv(dim_pred_path)[
                ["latitude", "longitude"] + [c for c in _DIMENSION_COLS]
            ]
            pred_table = pred_table.merge(
                dim_preds, on=["latitude", "longitude"], how="left"
            )
            found_dims = [c for c in _DIMENSION_COLS if c in pred_table.columns]
            logger.info("Merged dimension predictions: %d columns", len(found_dims))
        except Exception as e:
            logger.warning("Could not merge dimension predictions (non-fatal): %s", e)
    else:
        logger.debug("Dimension predictions file not found — skipping merge: %s", dim_pred_path)

    # -----------------------------------------------------------------------
    # Spatial join: assign each grid cell to its LGA
    # -----------------------------------------------------------------------
    valid = pred_table.dropna(subset=["longitude", "latitude"]).copy()
    gdf = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid["longitude"], valid["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        gdf, lga[["lga_id", "state", "lga_name", "geometry"]],
        how="left", predicate="within",
    )
    n_matched = joined["lga_id"].notna().sum()
    logger.info(
        "Grid cells matched to LGA: %d / %d (%.1f%%)",
        n_matched, len(joined), n_matched / len(joined) * 100,
    )

    # -----------------------------------------------------------------------
    # Population-weighted aggregation
    # -----------------------------------------------------------------------
    pred_cols = [c for c in _PRED_COLS if c in joined.columns]
    dim_cols = [c for c in _DIMENSION_COLS if c in joined.columns]
    theme_cols = [c for c in joined.columns if c.startswith("ridge_theme__")]
    # Per-feature β·z contribution columns (from prediction_breakdown merge)
    bdg_cols = [c for c in joined.columns if c.startswith("ridge_bdg__") and not c.endswith("_popup")]
    # Raw feature value columns (raw__<feature>)
    raw_cols = [c for c in joined.columns if c.startswith("raw__")]
    lga_matched = joined.dropna(subset=["lga_id"]).copy()

    rows = []
    for lga_id_val, grp in lga_matched.groupby("lga_id"):
        pop = grp["population"].fillna(0).clip(lower=0).values
        pop_sum = pop.sum()

        row: dict = {
            "lga_id":           lga_id_val,
            "state":            grp["state"].iloc[0],
            "lga_name":         grp["lga_name"].iloc[0],
            "n_cells":          len(grp),
            "total_population": int(pop_sum),
            "pct_urban":        round(
                float((grp["is_urban"].fillna(0).values * pop).sum() / max(pop_sum, 1)) * 100, 1
            ),
            "mics_state_truth": round(float(grp["moderate_prevalence"].iloc[0]), 2)
                                 if "moderate_prevalence" in grp.columns else None,
        }
        for col in pred_cols:
            vals = grp[col].fillna(0).values
            row[col] = round(
                float(np.average(vals, weights=pop)) if pop_sum > 0 else float(vals.mean()),
                2,
            )
        def _popw_mean(col):
            vals = grp[col].values.astype(float)
            okw = np.isfinite(vals) & (pop > 0)
            if okw.any() and pop[okw].sum() > 0:
                return round(float(np.average(vals[okw], weights=pop[okw])), 4)
            elif np.isfinite(vals).any():
                return round(float(np.nanmean(vals)), 4)
            return np.nan

        for col in dim_cols:
            row[col] = _popw_mean(col)

        for col in theme_cols:
            row[col] = _popw_mean(col)

        for col in bdg_cols:
            row[col] = _popw_mean(col)

        for col in raw_cols:
            row[col] = _popw_mean(col)

        rows.append(row)

    lga_preds = pd.DataFrame(rows)
    logger.info("LGAs with predictions: %d / %d", len(lga_preds), len(lga))

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    lga_preds.to_csv(csv_path, index=False)
    logger.info("LGA predictions CSV saved: %s", csv_path)

    # -----------------------------------------------------------------------
    # Merge geometry and save GeoJSON
    # -----------------------------------------------------------------------
    # GeoJSON: omit raw__ and ridge_bdg__ columns (they balloon file size);
    # the CSV contains the full set.
    keep_geo_cols = (
        ["lga_id", "state", "lga_name", "n_cells", "total_population",
         "pct_urban", "mics_state_truth"]
        + pred_cols
        + [c for c in dim_cols if c in lga_preds.columns]
        + [c for c in theme_cols if c in lga_preds.columns]
    )
    keep_geo_cols = [c for c in keep_geo_cols if c in lga_preds.columns]
    lga_geo = lga.merge(
        lga_preds[keep_geo_cols].drop(columns=["state", "lga_name"], errors="ignore"),
        on="lga_id",
        how="left",
    )
    lga_geo.to_file(geojson_path, driver="GeoJSON")
    logger.info("LGA GeoJSON saved: %s", geojson_path)

    # -----------------------------------------------------------------------
    # Quick diagnostic: top 10 most deprived LGAs
    # -----------------------------------------------------------------------
    best_col = next((c for c in ["gbm_moderate", "ridge_moderate"] if c in lga_preds.columns), None)
    if best_col:
        top10 = lga_preds.nlargest(10, best_col)[
            ["state", "lga_name", "total_population", "pct_urban",
             "mics_state_truth", best_col]
        ]
        logger.info("Top 10 most deprived LGAs (%s):\n%s", best_col, top10.to_string(index=False))

    return csv_path, geojson_path


if __name__ == "__main__":
    from src.utils.config_loader import load_config, setup_logging

    _cfg = load_config("config/config_nga.yaml")
    setup_logging(_cfg)

    _pred = pd.read_parquet(_cfg["paths"]["predictions_parquet"]
                            if "predictions_parquet" in _cfg["paths"]
                            else "Data/outputs/nga/tables/nga_predictions.parquet")
    aggregate_to_lga(_cfg, _pred)
