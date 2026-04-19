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
from scipy.stats import norm

from src.utils.config_loader import load_config, setup_logging, get_available_features

logger = logging.getLogger(__name__)


def _assign_quintile_memberships(
    df: pd.DataFrame,
    quintile_targets: pd.DataFrame,
    zone_col: str = "subregion",
) -> pd.DataFrame:
    """
    Compute soft quintile membership probabilities and quintile-weighted pseudo-targets.

    For each cell, computes its within-zone RWI percentile, maps that to
    quintile probabilities using a smooth kernel, then produces a continuous
    pseudo-target as the weighted average of quintile prevalences.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling table with RWI and subregion columns.
    quintile_targets : pd.DataFrame
        Quintile target table with columns: subregion, quintile, moderate_prevalence,
        severe_prevalence.
    zone_col : str
        Column name for zone assignment.

    Returns
    -------
    pd.DataFrame
        Input dataframe with added columns: p_q1..p_q5, quintile_target_moderate,
        quintile_target_severe.
    """
    df = df.copy()

    # Quintile boundaries (percentile cutoffs: 0-20, 20-40, 40-60, 60-80, 80-100)
    quintile_centers = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    quintile_names = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    bandwidth = 0.15  # controls smoothness of quintile membership

    # Initialize columns
    for q in quintile_names:
        df[f"p_{q.lower()}"] = 0.0
    df["quintile_target_moderate"] = np.nan
    df["quintile_target_severe"] = np.nan

    zones = df[zone_col].unique()
    zones = [z for z in zones if z != "Unknown"]

    for zone in zones:
        zone_mask = df[zone_col] == zone
        rwi_vals = df.loc[zone_mask, "rwi"].values

        if len(rwi_vals) == 0:
            continue

        # Compute within-zone percentile (0=poorest, 1=richest)
        # rank(pct=True) gives low percentile to low RWI (poor) cells
        # Q1 center=0.1 (poorest), Q5 center=0.9 (richest)
        # So low RWI → low pctile → near Q1 center → high P(Q1) → high deprivation
        pctile = pd.Series(rwi_vals).rank(pct=True).values

        # Compute soft quintile membership using Gaussian kernel
        probs = np.zeros((len(rwi_vals), 5))
        for qi, center in enumerate(quintile_centers):
            probs[:, qi] = norm.pdf(pctile, loc=center, scale=bandwidth)

        # Normalize to sum to 1
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums

        for qi, qname in enumerate(quintile_names):
            df.loc[zone_mask, f"p_{qname.lower()}"] = probs[:, qi]

        # Look up quintile prevalences for this zone
        zone_qt = quintile_targets[quintile_targets["subregion"] == zone]
        if len(zone_qt) == 0:
            # Fall back to national (_T) quintile targets
            zone_qt = quintile_targets[quintile_targets["subregion"] == "_T"]

        if len(zone_qt) == 0:
            logger.warning("No quintile targets found for zone '%s'. Skipping.", zone)
            continue

        # Build prevalence vectors
        mod_prev = np.zeros(5)
        sev_prev = np.zeros(5)
        for qi, qname in enumerate(quintile_names):
            q_row = zone_qt[zone_qt["quintile"] == qname]
            if len(q_row) > 0:
                mod_prev[qi] = q_row["moderate_prevalence"].iloc[0]
                sev_prev[qi] = q_row["severe_prevalence"].iloc[0]

        # Compute weighted pseudo-target: sum(P(Qk) * prevalence(Qk))
        qt_mod = probs @ mod_prev
        qt_sev = probs @ sev_prev
        df.loc[zone_mask, "quintile_target_moderate"] = qt_mod
        df.loc[zone_mask, "quintile_target_severe"] = qt_sev

    # Log summary
    valid = df["quintile_target_moderate"].notna()
    if valid.sum() > 0:
        logger.info(
            "Quintile pseudo-targets assigned to %d cells. "
            "Moderate range: [%.1f%%, %.1f%%], mean=%.1f%%",
            valid.sum(),
            df.loc[valid, "quintile_target_moderate"].min(),
            df.loc[valid, "quintile_target_moderate"].max(),
            df.loc[valid, "quintile_target_moderate"].mean(),
        )
        # Verify probabilities sum to 1
        p_cols = [f"p_{q.lower()}" for q in quintile_names]
        p_sums = df.loc[valid, p_cols].sum(axis=1)
        logger.info(
            "Quintile probability sums: min=%.4f, max=%.4f (should be ~1.0)",
            p_sums.min(), p_sums.max(),
        )

    return df


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

    # Filter targets to subnational rows only (exclude national _T if present)
    national_code = cfg["targets"].get("subregion_national")
    if national_code:
        subnational_targets = targets[targets["subregion"] != national_code].copy()
    else:
        subnational_targets = targets.copy()

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
    # (spatial imputation in step02 should have filled most; fill remaining with 0)
    if "population" in merged.columns:
        n_pop_missing = merged["population"].isna().sum()
        if n_pop_missing > 0:
            logger.info(
                "Filling %d remaining missing population values with 0 "
                "(after spatial imputation, these are beyond max_distance).",
                n_pop_missing,
            )
            # Mark these as imputed too
            if "population_imputed" not in merged.columns:
                merged["population_imputed"] = 0
            merged.loc[merged["population"].isna(), "population_imputed"] = 1
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
    feature_cols = get_available_features(cfg, merged)
    for col in feature_cols:
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

    # ------------------------------------------------------------------
    # New enrichment features (Nigeria only) — nightlights, schools, health,
    # conflict, rainfall — pre-computed in Data/Nigeria/features/
    # ------------------------------------------------------------------
    country_code = cfg.get("country", {}).get("code", "")
    new_feats_path = cfg.get("paths", {}).get("new_features_parquet", "")
    if country_code == "NGA" and new_feats_path and os.path.isfile(new_feats_path):
        try:
            new_feats = pd.read_parquet(new_feats_path)
            # Join on longitude + latitude (rounded to avoid float drift)
            key_cols = ["longitude", "latitude"]
            for c in key_cols:
                merged[c] = merged[c].round(6)
                new_feats[c] = new_feats[c].round(6)
            feat_cols = [c for c in new_feats.columns if c not in key_cols and c not in merged.columns]
            merged = merged.merge(new_feats[key_cols + feat_cols], on=key_cols, how="left")
            logger.info(
                "New enrichment features joined: %s — %d cells matched.",
                feat_cols, merged[feat_cols[0]].notna().sum() if feat_cols else 0,
            )
        except Exception as e:
            logger.warning("Could not join new enrichment features: %s", e)

    # ------------------------------------------------------------------
    # Hierarchy columns (Nigeria only) — needed for hierarchical validation
    # ------------------------------------------------------------------
    country_code = cfg.get("country", {}).get("code", "")
    if country_code == "NGA":
        try:
            from src.utils.admin_mappings import add_geopolitical_zones, add_state_urban_rural
            merged = add_geopolitical_zones(merged, zone_col="subregion")
            merged = add_state_urban_rural(merged, state_col="subregion", urban_col="is_urban")
            logger.info("Nigeria hierarchy columns added: geopolitical_zone, state_urban_rural.")
        except Exception as e:
            logger.warning("Could not add Nigeria hierarchy columns: %s", e)

    # ------------------------------------------------------------------
    # Quintile-based pseudo-targets (if available)
    # ------------------------------------------------------------------
    output_prefix = cfg.get("country", {}).get("output_prefix", "jam")
    quintile_path = os.path.join(
        os.path.dirname(cfg["paths"]["targets_file"]),
        f"{output_prefix}_quintile_targets.csv",
    )
    if cfg["targets"].get("use_quintile_targets", False) and os.path.isfile(quintile_path):
        logger.info("Loading quintile targets from: %s", quintile_path)
        quintile_targets = pd.read_csv(quintile_path)
        zone_col = cfg["modeling"]["admin_zone_col"]
        merged = _assign_quintile_memberships(merged, quintile_targets, zone_col)
    else:
        logger.info("Quintile targets not available or disabled. Skipping quintile membership.")

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
