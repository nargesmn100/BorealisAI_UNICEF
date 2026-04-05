"""
RWI-Based Redistribution Baseline

The current operational baseline used by UNICEF and similar organisations:
redistribute official zone-level deprivation totals across space using the
Relative Wealth Index (RWI) as the spatial allocation weight.

Scientific interpretation
--------------------------
This baseline assumes that *lower* RWI (poorer areas) should receive
*higher* deprivation scores.  Concretely, within each admin zone we invert
and scale the RWI to produce allocation weights, then rescale so the
population-weighted mean matches the official zone target.

Two sub-strategies are supported:
1. ``inverted_rwi``: weight = exp(-rwi) — continuous smooth allocation
2. ``rank_rwi``:     weight based on rank(rwi) descending — rank-based

Design choices
--------------
- RWI is inverted because higher RWI = wealthier = less deprived.
- exp(-rwi) is used rather than (-rwi) to ensure all weights are positive.
- Population weighting is applied so the reconciliation is consistent
  with a population-weighted zone prevalence interpretation.

Output columns added:
  rwi_moderate_raw         : raw (pre-reconciliation) RWI-based score
  rwi_moderate             : reconciled RWI-based moderate prevalence
  rwi_severe               : reconciled RWI-based severe prevalence
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.reconciliation.admin_reconcile import reconcile_predictions

logger = logging.getLogger(__name__)

RwiStrategy = Literal["inverted_rwi", "rank_rwi"]


def _compute_rwi_raw_score(
    rwi: np.ndarray,
    strategy: RwiStrategy = "inverted_rwi",
) -> np.ndarray:
    """
    Convert RWI values into raw deprivation proxy scores.

    Lower RWI → higher deprivation proxy.

    Parameters
    ----------
    rwi : np.ndarray
    strategy : str

    Returns
    -------
    np.ndarray
        Positive weights (relative deprivation proxy scores).
    """
    if strategy == "inverted_rwi":
        # exp(-rwi) maps high RWI (wealthy) to low score, low RWI (poor) to high score
        scores = np.exp(-rwi)
    elif strategy == "rank_rwi":
        # Rank from highest-deprivation (lowest RWI) to lowest-deprivation
        n = len(rwi)
        ranks = pd.Series(rwi).rank(method="average", ascending=True).values
        scores = (n - ranks + 1).astype(float)  # invert so rank 1 (lowest RWI) = max score
    else:
        raise ValueError(f"Unknown RWI strategy: '{strategy}'")

    return scores


def apply_rwi_baseline(
    df: pd.DataFrame,
    zone_col: str = "subregion",
    rwi_col: str = "rwi",
    population_col: str = "population",
    moderate_target_col: str = "moderate_prevalence",
    severe_target_col: str = "severe_prevalence",
    strategy: RwiStrategy = "inverted_rwi",
) -> pd.DataFrame:
    """
    Apply RWI-based redistribution baseline.

    Steps:
    1. Compute raw RWI-derived deprivation scores (inverted RWI)
    2. Reconcile moderate prevalence predictions to zone targets
    3. Reconcile severe prevalence predictions to zone targets

    Parameters
    ----------
    df : pd.DataFrame
        Modeling table.
    zone_col : str
    rwi_col : str
    population_col : str
    moderate_target_col, severe_target_col : str
    strategy : RwiStrategy

    Returns
    -------
    pd.DataFrame
        With RWI baseline columns added.
    """
    result = df.copy()

    valid_mask = (
        result[zone_col].notna()
        & (result[zone_col] != "Unknown")
        & result[rwi_col].notna()
        & result[moderate_target_col].notna()
    )

    logger.info(
        "Applying RWI baseline (strategy='%s') to %d valid cells.",
        strategy, valid_mask.sum(),
    )

    # ------------------------------------------------------------------
    # Compute raw RWI-based deprivation scores per zone
    # This is done within-zone to preserve the within-zone signal
    # ------------------------------------------------------------------
    result["rwi_raw_score"] = np.nan

    zones = result.loc[valid_mask, zone_col].unique()

    for zone in sorted(zones):
        zmask = valid_mask & (result[zone_col] == zone)
        rwi_vals = result.loc[zmask, rwi_col].values.astype(float)

        # Fill any NaN in RWI with zone median
        rwi_median = np.nanmedian(rwi_vals)
        rwi_vals = np.where(np.isnan(rwi_vals), rwi_median, rwi_vals)

        raw_scores = _compute_rwi_raw_score(rwi_vals, strategy=strategy)
        result.loc[zmask, "rwi_raw_score"] = raw_scores

        logger.debug(
            "Zone '%s': %d cells, raw score range [%.4f, %.4f]",
            zone, zmask.sum(), raw_scores.min(), raw_scores.max(),
        )

    # ------------------------------------------------------------------
    # Reconcile raw scores to moderate prevalence targets
    # ------------------------------------------------------------------
    logger.info("Reconciling RWI baseline to moderate prevalence targets...")
    result = reconcile_predictions(
        result,
        raw_score_col="rwi_raw_score",
        target_col=moderate_target_col,
        zone_col=zone_col,
        population_col=population_col,
        output_col="rwi_moderate",
        strategy="population_weighted",
    )

    # ------------------------------------------------------------------
    # Reconcile to severe prevalence targets
    # ------------------------------------------------------------------
    logger.info("Reconciling RWI baseline to severe prevalence targets...")
    result = reconcile_predictions(
        result,
        raw_score_col="rwi_raw_score",
        target_col=severe_target_col,
        zone_col=zone_col,
        population_col=population_col,
        output_col="rwi_severe",
        strategy="population_weighted",
    )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    for col, label in [("rwi_moderate", "Moderate"), ("rwi_severe", "Severe")]:
        vals = result.loc[valid_mask, col]
        logger.info(
            "RWI baseline — %s: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            label, vals.mean(), vals.std(), vals.min(), vals.max(),
        )

    return result


def run(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Entry point for the RWI redistribution baseline.

    Parameters
    ----------
    cfg : dict
    df : pd.DataFrame
        Modeling table.

    Returns
    -------
    pd.DataFrame
    """
    zone_col = cfg["modeling"]["admin_zone_col"]
    # Use the renamed column names from the modeling table (not raw Excel names)
    moderate_col = "moderate_prevalence"
    severe_col = "severe_prevalence"

    return apply_rwi_baseline(
        df,
        zone_col=zone_col,
        moderate_target_col=moderate_col,
        severe_target_col=severe_col,
        strategy="inverted_rwi",
    )
