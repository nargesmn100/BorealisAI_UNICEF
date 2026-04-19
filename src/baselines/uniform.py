"""
Uniform Allocation Baseline

The simplest possible baseline: assign every grid cell within a zone
the same deprivation score — the zone-level official prevalence.

Scientific interpretation
--------------------------
This baseline assumes that deprivation is uniformly distributed across
space within each administrative zone.  It provides zero spatial information
beyond what is already in the official statistics.

Any model that cannot beat this baseline adds no spatial information.

Output columns added to DataFrame:
  uniform_moderate   : uniform allocation for moderate prevalence
  uniform_severe     : uniform allocation for severe prevalence
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_uniform_baseline(
    df: pd.DataFrame,
    zone_col: str = "subregion",
    moderate_target_col: str = "moderate_prevalence",
    severe_target_col: str = "severe_prevalence",
    moderate_output_col: str = "uniform_moderate",
    severe_output_col: str = "uniform_severe",
) -> pd.DataFrame:
    """
    Assign each cell the zone-level official prevalence (uniform allocation).

    Parameters
    ----------
    df : pd.DataFrame
        Modeling table with zone assignment and target columns.
    zone_col : str
        Column identifying the admin zone.
    moderate_target_col : str
        Column with official moderate poverty prevalence (%).
    severe_target_col : str
        Column with official severe poverty prevalence (%).
    moderate_output_col, severe_output_col : str
        Output column names.

    Returns
    -------
    pd.DataFrame
        Copy with uniform allocation columns added.
    """
    result = df.copy()
    result[moderate_output_col] = np.nan
    result[severe_output_col] = np.nan

    valid_mask = (
        result[zone_col].notna()
        & (result[zone_col] != "Unknown")
        & result[moderate_target_col].notna()
    )

    logger.info(
        "Applying uniform baseline to %d valid cells (%d total).",
        valid_mask.sum(), len(result),
    )

    # Each cell gets exactly the zone target — this is the trivial allocation
    result.loc[valid_mask, moderate_output_col] = (
        result.loc[valid_mask, moderate_target_col].values
    )
    result.loc[valid_mask, severe_output_col] = (
        result.loc[valid_mask, severe_target_col].values
    )

    # Summarise
    for col, label in [
        (moderate_output_col, "Moderate"),
        (severe_output_col, "Severe"),
    ]:
        logger.info(
            "Uniform baseline — %s: mean=%.3f, std=%.3f (across zones)",
            label,
            result.loc[valid_mask, col].mean(),
            result.loc[valid_mask, col].std(),
        )

    return result


def run(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Entry point for the uniform baseline.

    Parameters
    ----------
    cfg : dict
    df : pd.DataFrame
        Modeling table.

    Returns
    -------
    pd.DataFrame
        Modeling table with uniform baseline predictions.
    """
    zone_col = cfg["modeling"]["admin_zone_col"]
    # Use the renamed column names from the modeling table (not raw Excel names)
    moderate_col = "moderate_prevalence"
    severe_col = "severe_prevalence"

    df = apply_uniform_baseline(
        df,
        zone_col=zone_col,
        moderate_target_col=moderate_col,
        severe_target_col=severe_col,
    )

    # Depth metrics — same uniform assignment
    if "moderate_depth" in df.columns:
        df = apply_uniform_baseline(
            df,
            zone_col=zone_col,
            moderate_target_col="moderate_depth",
            severe_target_col="severe_depth",
            moderate_output_col="uniform_moderate_depth",
            severe_output_col="uniform_severe_depth",
        )

    return df
