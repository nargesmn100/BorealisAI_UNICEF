"""
Administrative Reconciliation Module

Implements hard post-processing reconciliation so that grid-level predictions
exactly match official administrative totals within each zone.

Scientific framing
------------------
The model predicts a raw deprivation *score* for each grid cell.
These raw scores encode relative spatial variation but are not calibrated
to official prevalence values.

Post-processing rescales each cell's score *within its admin zone* so that
the population-weighted average of final predictions equals the official
zone-level prevalence.

Two reconciliation strategies
------------------------------
1. ``proportional_rescale`` (default, recommended):
   Each cell score is multiplied by a zone-specific scale factor:

       scale_k = target_k / mean(raw_scores_in_zone_k)

   This preserves the *relative* ordering of predictions within each zone
   while forcing the zone average to the official target.

2. ``population_weighted_rescale``:
   Enforces the constraint using a population-weighted mean:

       scale_k = target_k / weighted_mean(raw_scores, population)

   This is more appropriate when the official statistic is interpreted as
   a population-weighted prevalence.

Constraints and guarantees
---------------------------
- The zone-level weighted mean of final predictions equals the official target.
- No individual cell prediction can be forced below 0 or above 100.
  (Clamping is applied and logged as a warning if it occurs.)
- Cells with no valid raw prediction are excluded from the mean calculation.

Usage
-----
    from src.reconciliation.admin_reconcile import reconcile_predictions
    reconciled = reconcile_predictions(
        df=modeling_table,
        raw_score_col="raw_score",
        target_col="moderate_prevalence",
        zone_col="subregion",
        population_col="population",
        strategy="population_weighted",
    )
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ReconcileStrategy = Literal["proportional", "population_weighted"]


def reconcile_predictions(
    df: pd.DataFrame,
    raw_score_col: str,
    target_col: str,
    zone_col: str = "subregion",
    population_col: str = "population",
    output_col: str | None = None,
    strategy: ReconcileStrategy = "population_weighted",
    clamp_min: float = 0.0,
    clamp_max: float = 100.0,
) -> pd.DataFrame:
    """
    Rescale raw grid-level predictions to match official zone-level targets.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling table with raw scores and official targets.
    raw_score_col : str
        Column containing raw model predictions.
    target_col : str
        Column containing official admin-level target (e.g. "moderate_prevalence").
    zone_col : str
        Column identifying the reconciliation zone (e.g. "subregion").
    population_col : str
        Column with population counts for population-weighted strategy.
    output_col : str or None
        Name of the output column.  Defaults to ``raw_score_col + "_reconciled"``.
    strategy : {"proportional", "population_weighted"}
        Reconciliation strategy.
    clamp_min, clamp_max : float
        Min/max bounds for final predictions (default 0–100%).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the reconciled predictions column added.
    """
    if output_col is None:
        output_col = f"{raw_score_col}_reconciled"

    result = df.copy()
    
    # Only initialize output column if it's different from raw_score_col
    if output_col != raw_score_col:
        result[output_col] = np.nan

    # Work only on rows with valid raw scores and targets
    valid_mask = (
        result[raw_score_col].notna()
        & result[target_col].notna()
        & result[zone_col].notna()
        & (result[zone_col] != "Unknown")
    )

    logger.info(
        "Reconciling predictions using strategy='%s'. "
        "Valid rows: %d / %d",
        strategy, valid_mask.sum(), len(result),
    )

    zones = result.loc[valid_mask, zone_col].unique()

    for zone in sorted(zones):
        zone_mask = valid_mask & (result[zone_col] == zone)
        n_cells = zone_mask.sum()

        raw_scores = result.loc[zone_mask, raw_score_col].values
        target_value = result.loc[zone_mask, target_col].iloc[0]

        if n_cells == 0:
            logger.warning("Zone '%s': no valid cells. Skipping.", zone)
            continue

        if strategy == "proportional":
            zone_mean = raw_scores.mean()
        elif strategy == "population_weighted":
            pop = result.loc[zone_mask, population_col].values.astype(float)
            pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)
            total_pop = pop.sum()
            if total_pop <= 0:
                logger.warning(
                    "Zone '%s': total population = 0. Falling back to "
                    "simple mean for reconciliation.",
                    zone,
                )
                zone_mean = raw_scores.mean()
            else:
                zone_mean = np.average(raw_scores, weights=pop)
        else:
            raise ValueError(
                f"Unknown reconciliation strategy: '{strategy}'. "
                "Choose 'proportional' or 'population_weighted'."
            )

        if zone_mean <= 0:
            logger.warning(
                "Zone '%s': zone mean score = %.4f (non-positive). "
                "Cannot scale to target=%.2f. "
                "Setting all cells to uniform target value.",
                zone, zone_mean, target_value,
            )
            scale_factor = None
        else:
            scale_factor = target_value / zone_mean

        if scale_factor is not None:
            reconciled = raw_scores * scale_factor
        else:
            reconciled = np.full(n_cells, target_value)

        # Clamp to valid range
        n_below = (reconciled < clamp_min).sum()
        n_above = (reconciled > clamp_max).sum()
        if n_below > 0 or n_above > 0:
            logger.warning(
                "Zone '%s': clamping %d cells below %.1f and %d cells above %.1f.",
                zone, n_below, clamp_min, n_above, clamp_max,
            )
        reconciled = np.clip(reconciled, clamp_min, clamp_max)

        result.loc[zone_mask, output_col] = reconciled

        # Verify reconciliation accuracy
        if strategy == "population_weighted":
            pop = result.loc[zone_mask, population_col].values.astype(float)
            pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)
            achieved = np.average(reconciled, weights=pop) if pop.sum() > 0 else reconciled.mean()
        else:
            achieved = reconciled.mean()

        logger.info(
            "Zone %-45s | target=%.3f%% | achieved=%.3f%% | "
            "scale=%.4f | cells=%d",
            f"'{zone}'",
            target_value, achieved,
            scale_factor if scale_factor is not None else 1.0,
            n_cells,
        )

    return result


def reconcile_uncertainty_bounds(
    df: pd.DataFrame,
    lower_col: str,
    upper_col: str,
    raw_col: str,
    reconciled_col: str,
    zone_col: str = "subregion",
    population_col: str = "population",
    clamp_min: float = 0.0,
    clamp_max: float = 100.0,
) -> pd.DataFrame:
    """
    Propagate uncertainty bounds through reconciliation by applying the same
    per-zone scale factor used for point predictions.

    For each zone, the scale factor is: reconciled / raw (population-weighted).
    This factor is applied to lower and upper CI bounds, then clamped to [0, 100].

    Parameters
    ----------
    df : pd.DataFrame
    lower_col, upper_col : str
        Columns with raw CI bounds.
    raw_col : str
        Column with raw (pre-reconciliation) point predictions.
    reconciled_col : str
        Column with reconciled point predictions.
    zone_col : str
    population_col : str
    clamp_min, clamp_max : float

    Returns
    -------
    pd.DataFrame
        Copy of df with lower_col and upper_col updated to reconciled values.
    """
    result = df.copy()

    valid_mask = (
        result[raw_col].notna()
        & result[reconciled_col].notna()
        & result[lower_col].notna()
        & result[upper_col].notna()
        & result[zone_col].notna()
        & (result[zone_col] != "Unknown")
    )

    zones = result.loc[valid_mask, zone_col].unique()

    for zone in sorted(zones):
        zmask = valid_mask & (result[zone_col] == zone)
        raw_vals = result.loc[zmask, raw_col].values.astype(float)
        pop = result.loc[zmask, population_col].values.astype(float)
        pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)

        total_pop = pop.sum()
        if total_pop > 0:
            raw_mean = np.average(raw_vals, weights=pop)
        else:
            raw_mean = raw_vals.mean()

        if raw_mean <= 0:
            continue

        recon_vals = result.loc[zmask, reconciled_col].values.astype(float)
        if total_pop > 0:
            recon_mean = np.average(recon_vals, weights=pop)
        else:
            recon_mean = recon_vals.mean()

        scale_factor = recon_mean / raw_mean

        result.loc[zmask, lower_col] = np.clip(
            result.loc[zmask, lower_col].values * scale_factor, clamp_min, clamp_max,
        )
        result.loc[zmask, upper_col] = np.clip(
            result.loc[zmask, upper_col].values * scale_factor, clamp_min, clamp_max,
        )

    logger.info(
        "Uncertainty bounds (%s, %s) propagated through reconciliation.",
        lower_col, upper_col,
    )
    return result


def verify_reconciliation(
    df: pd.DataFrame,
    reconciled_col: str,
    target_col: str,
    zone_col: str,
    population_col: str,
    tolerance: float = 0.01,
) -> bool:
    """
    Verify that reconciled predictions match official targets within tolerance.

    Parameters
    ----------
    df : pd.DataFrame
    reconciled_col : str
    target_col : str
    zone_col : str
    population_col : str
    tolerance : float
        Maximum allowed absolute difference (in percentage points).

    Returns
    -------
    bool
        True if all zones are within tolerance.
    """
    all_ok = True
    valid = df[reconciled_col].notna() & df[target_col].notna()
    zones = df.loc[valid, zone_col].unique()

    logger.info("=== Reconciliation Verification (tolerance=%.4f pp) ===", tolerance)

    for zone in sorted(zones):
        zmask = valid & (df[zone_col] == zone)
        target = df.loc[zmask, target_col].iloc[0]
        pop = df.loc[zmask, population_col].values.astype(float)
        pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)
        recon = df.loc[zmask, reconciled_col].values

        if pop.sum() > 0:
            achieved = np.average(recon, weights=pop)
        else:
            achieved = recon.mean()

        diff = abs(achieved - target)
        status = "✓ OK" if diff <= tolerance else "✗ FAIL"
        logger.info(
            "  Zone %-45s | target=%.4f | achieved=%.4f | diff=%.6f | %s",
            f"'{zone}'", target, achieved, diff, status,
        )
        if diff > tolerance:
            all_ok = False

    return all_ok
