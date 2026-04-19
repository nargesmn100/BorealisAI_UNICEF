"""
Heuristic Baseline: Manual-Style Wealth-Based Redistribution
==============================================================

This baseline represents a plausible current practice for spatial
disaggregation: within each zone, assign different poverty rates based
on wealth terciles using simple multipliers.

This is more sophisticated than uniform allocation but simpler than
RWI redistribution, representing a manual analyst approach.

Methodology
-----------
Within each zone:
    - Bottom tercile (poorest 1/3): zone_rate × 1.3
    - Middle tercile: zone_rate × 1.0
    - Top tercile (richest 1/3): zone_rate × 0.7

Then reconcile to match exact zone totals via population-weighted rescaling.

This provides a baseline between:
    - Uniform (no spatial variation)
    - RWI redistribution (data-driven)
    - ML models (complex algorithms)
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _apply_tercile_heuristic(
    df: pd.DataFrame,
    zone_mask: np.ndarray,
    target_col: str,
    rwi_col: str = 'rwi'
) -> np.ndarray:
    """
    Apply tercile-based heuristic within a zone.
    
    Parameters
    ----------
    df : pd.DataFrame
        Modeling table
    zone_mask : np.ndarray
        Boolean mask for cells in this zone
    target_col : str
        Target column (e.g., 'moderate_prevalence')
    rwi_col : str
        RWI column for ranking
    
    Returns
    -------
    heuristic_scores : np.ndarray
        Scores for cells in this zone
    """
    zone_df = df[zone_mask].copy()
    
    if len(zone_df) == 0:
        return np.array([])
    
    # Get zone-level official rate
    zone_rate = zone_df[target_col].iloc[0]
    
    # Rank cells by RWI (lower RWI = poorer)
    # Use qcut to create 3 equal-sized terciles
    try:
        rwi_terciles = pd.qcut(
            zone_df[rwi_col],
            q=3,
            labels=['bottom', 'middle', 'top'],
            duplicates='drop'  # Handle ties
        )
    except ValueError:
        # If all values are identical, treat as middle tercile
        rwi_terciles = pd.Series(['middle'] * len(zone_df), index=zone_df.index)
    
    # Define multipliers
    # Bottom tercile (poorest): higher poverty
    # Top tercile (richest): lower poverty
    multipliers = {
        'bottom': 1.3,   # 30% above zone average
        'middle': 1.0,   # Zone average
        'top': 0.7       # 30% below zone average
    }
    
    # Apply multipliers - convert categorical to numeric first
    heuristic_scores = rwi_terciles.map(multipliers).astype(float) * zone_rate
    
    return heuristic_scores.values


def run(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply heuristic baseline: wealth tercile-based redistribution.
    
    This represents a manual-style approach that a policy analyst might use:
    "Rich areas have lower poverty, poor areas have higher poverty" with
    simple multipliers, then reconcile to official zone totals.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    df : pd.DataFrame
        Modeling table with RWI and target columns
    
    Returns
    -------
    df : pd.DataFrame
        Updated with heuristic_moderate, heuristic_severe, etc. columns
    """
    from src.reconciliation.admin_reconcile import reconcile_predictions
    
    logger.info("Heuristic Baseline: Wealth Tercile-Based Redistribution")
    logger.info("Method: Within-zone tercile multipliers (bottom=1.3x, middle=1.0x, top=0.7x)")
    
    # Only model cells with full data
    mask = df['in_modeling_sample'].fillna(False)
    
    # Process prevalence targets
    for target_type in ['moderate', 'severe']:
        target_col = f'{target_type}_prevalence'
        out_col = f'heuristic_{target_type}'
        
        logger.info(f"--- Heuristic {target_type} prevalence ---")
        
        # Initialize with NaN
        df[out_col] = np.nan
        
        # For each zone, apply tercile multipliers
        for zone in df['subregion'].dropna().unique():
            zone_mask = mask & (df['subregion'] == zone)
            
            if zone_mask.sum() == 0:
                continue
            
            # Apply heuristic
            heuristic_scores = _apply_tercile_heuristic(
                df, zone_mask, target_col
            )
            
            df.loc[zone_mask, out_col] = heuristic_scores
            
            logger.info(
                f"  Zone '{zone}': {zone_mask.sum():,} cells, "
                f"range [{heuristic_scores.min():.1f}%, {heuristic_scores.max():.1f}%]"
            )
        
        # Reconcile to exact zone totals
        logger.info(f"Reconciling heuristic {target_type} to official zone targets...")
        df = reconcile_predictions(
            df,
            raw_score_col=out_col,
            target_col=target_col,
            output_col=out_col,
            strategy='population_weighted'
        )
    
    # Process depth targets
    for target_type in ['moderate', 'severe']:
        depth_col = f'{target_type}_depth'
        out_col = f'heuristic_{target_type}_depth'
        
        logger.info(f"--- Heuristic {target_type} depth ---")
        
        # Initialize with NaN
        df[out_col] = np.nan
        
        # For each zone, apply tercile multipliers
        for zone in df['subregion'].dropna().unique():
            zone_mask = mask & (df['subregion'] == zone)
            
            if zone_mask.sum() == 0:
                continue
            
            # Apply heuristic
            heuristic_scores = _apply_tercile_heuristic(
                df, zone_mask, depth_col
            )
            
            df.loc[zone_mask, out_col] = heuristic_scores
        
        # Reconcile to exact zone totals
        df = reconcile_predictions(
            df,
            raw_score_col=out_col,
            target_col=depth_col,
            output_col=out_col,
            strategy='population_weighted'
        )
        
        logger.info(f"Heuristic {target_type}_depth reconciled.")
    
    logger.info("Heuristic baseline complete.\n")
    
    return df
