"""
Step 05 — Merge Features and Targets into the Modeling Table

Joins the grid (with proxies + admin assignment) to the poverty targets.
Each grid point inherits the poverty prevalence value of its subregion.

This is the final table used for:
  - Baseline methods (uniform allocation, RWI redistribution)
  - ML model training / prediction
  - Evaluation

The merge strategy
------------------
Grid points with subregion ∈ {"Urban", "Rural", "Kingston Metropolitan Area (KMA)"}
get the corresponding subnational poverty target attached.

Points with subregion == "Unknown" (fell outside all parishes) are flagged
and excluded from modeling but retained in the table.

Output saved to:  cfg["paths"]["modeling_table_file"]   (Parquet)

Final modeling table columns
-----------------------------
  cell_id                   : unique grid cell identifier
  latitude, longitude       : WGS84 coordinates
  rwi                       : Relative Wealth Index score
  rwi_error                 : RWI posterior uncertainty
  population                : sampled child population
  log_population            : log1p(population)
  smod_class                : GHSL SMOD integer class
  smod_label                : SMOD human-readable label
  is_urban                  : 1=urban, 0=rural (from SMOD)
  travel_time_cities        : travel time to nearest city (minutes)
  travel_time_50k           : travel time to nearest 50k city (minutes)
  log_travel_time_cities    : log1p(travel_time_cities)
  log_travel_time_50k       : log1p(travel_time_50k)
  gid_1                     : GADM parish GID
  parish_name               : GADM parish name
  subregion                 : poverty subregion (Urban/Rural/KMA/Unknown)
  severe_prevalence         : admin-level severe child poverty (%)
  moderate_prevalence       : admin-level moderate child poverty (%)
  severe_depth              : admin-level severe poverty depth
  moderate_depth            : admin-level moderate poverty depth
  in_modeling_sample        : bool — True if point has valid admin target
"""

import logging
import os

import pandas as pd
import numpy as np

from src.utils.config_loader import load_config, setup_logging

logger = logging.getLogger(__name__)


def merge_features_and_targets(
    cfg: dict,
    grid_admin: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join the grid to the target table on the ``subregion`` column.

    Parameters
    ----------
    cfg : dict
        Loaded config.
    grid_admin : pd.DataFrame
        Grid with proxies and admin assignment (Step 03 output).
    targets : pd.DataFrame
        Clean target table (Step 04 output).

    Returns
    -------
    pd.DataFrame
        Modeling table.
    """
    # ------------------------------------------------------------------
    # Validate join keys
    # ------------------------------------------------------------------
    if "subregion" not in grid_admin.columns:
        raise ValueError(
            "'subregion' column missing from grid. "
            "Ensure Step 03 (assign_admin) has been run."
        )
    if "subregion" not in targets.columns:
        raise ValueError(
            "'subregion' column missing from targets. "
            "Ensure Step 04 (prepare_targets) has been run."
        )

    # Filter targets to subnational rows only (exclude national _T)
    national_code = cfg["targets"]["subregion_national"]
    subnational_targets = targets[targets["subregion"] != national_code].copy()

    subregions_in_grid = set(grid_admin["subregion"].unique())
    subregions_in_targets = set(subnational_targets["subregion"].unique())

    logger.info("Subregions in grid: %s", sorted(subregions_in_grid))
    logger.info("Subregions in targets: %s", sorted(subregions_in_targets))

    # Check for coverage gaps
    in_grid_not_targets = subregions_in_grid - subregions_in_targets - {"Unknown"}
    if in_grid_not_targets:
        logger.warning(
            "Subregions in grid but not in targets: %s. "
            "These cells will have no target attached.",
            in_grid_not_targets,
        )

    in_targets_not_grid = subregions_in_targets - subregions_in_grid
    if in_targets_not_grid:
        logger.warning(
            "Subregions in targets but not in grid: %s. "
            "These target values cannot be used for reconciliation.",
            in_targets_not_grid,
        )

    # ------------------------------------------------------------------
    # Perform the left join
    # ------------------------------------------------------------------
    target_cols = [
        "subregion",
        "severe_prevalence",
        "moderate_prevalence",
        "severe_depth",
        "moderate_depth",
    ]
    merged = grid_admin.merge(
        subnational_targets[target_cols],
        on="subregion",
        how="left",
    )

    # ------------------------------------------------------------------
    # Flag modeling sample
    # ------------------------------------------------------------------
    merged["in_modeling_sample"] = (
        merged["subregion"].isin(subregions_in_targets)
        & merged["moderate_prevalence"].notna()
    )

    n_total = len(merged)
    n_model = merged["in_modeling_sample"].sum()
    n_excluded = n_total - n_model
    logger.info(
        "Modeling table: %d total cells — %d in modeling sample, %d excluded "
        "(unknown subregion or missing target).",
        n_total, n_model, n_excluded,
    )

    # ------------------------------------------------------------------
    # Impute missing feature values
    # ------------------------------------------------------------------
    logger.info("Imputing missing feature values...")

    # Population: NaN means no population raster coverage → treat as 0
    # (usually coastal pixels or unpopulated wilderness)
    if "population" in merged.columns:
        n_pop_missing = merged["population"].isna().sum()
        if n_pop_missing > 0:
            logger.info(
                "Imputing %d missing population values with 0 "
                "(likely coastal/offshore pixels with no raster coverage).",
                n_pop_missing,
            )
            merged["population"] = merged["population"].fillna(0.0)
            merged["log_population"] = np.log1p(merged["population"])

    # Travel time: impute NaN with within-subregion median, then global median
    for col in ["travel_time_cities", "travel_time_50k",
                "log_travel_time_cities", "log_travel_time_50k"]:
        if col not in merged.columns:
            continue
        n_missing = merged[col].isna().sum()
        if n_missing > 0:
            # Try subregion median first
            merged[col] = merged.groupby("subregion")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # Fall back to global median for any remaining NaN
            global_median = merged[col].median()
            merged[col] = merged[col].fillna(global_median)
            logger.info(
                "Imputed %d missing values in '%s' with subregion/global median.",
                n_missing, col,
            )

    # ------------------------------------------------------------------
    # Feature completeness check
    # ------------------------------------------------------------------
    feature_cols = cfg["modeling"]["features"]
    for col in feature_cols:
        if col not in merged.columns:
            logger.warning(
                "Feature '%s' listed in config but not found in modeling table. "
                "Check pipeline steps.",
                col,
            )
        else:
            n_missing = merged.loc[merged["in_modeling_sample"], col].isna().sum()
            if n_missing > 0:
                logger.warning(
                    "Feature '%s': %d missing values in modeling sample after imputation.",
                    col, n_missing,
                )

    # ------------------------------------------------------------------
    # Summary statistics of targets
    # ------------------------------------------------------------------
    logger.info("Target value summary by subregion:")
    summary = (
        merged[merged["in_modeling_sample"]]
        .groupby("subregion")[["moderate_prevalence", "severe_prevalence"]]
        .first()
    )
    logger.info("\n%s", summary.to_string())

    return merged


def run(
    cfg: dict,
    grid_admin: pd.DataFrame | None = None,
    targets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Entry point for Step 05.

    Parameters
    ----------
    cfg : dict
    grid_admin : pd.DataFrame or None
    targets : pd.DataFrame or None

    Returns
    -------
    pd.DataFrame
        Final modeling table.
    """
    if grid_admin is None:
        path = cfg["paths"]["grid_with_admin_file"]
        logger.info("Loading grid with admin from: %s", path)
        grid_admin = pd.read_parquet(path)

    if targets is None:
        path = cfg["paths"]["targets_file"]
        logger.info("Loading targets from: %s", path)
        targets = pd.read_csv(path)

    table = merge_features_and_targets(cfg, grid_admin, targets)

    out_path = cfg["paths"]["modeling_table_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    table.to_parquet(out_path, index=False)
    logger.info("Modeling table saved to: %s  (%d rows)", out_path, len(table))

    return table


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    run(_cfg)
