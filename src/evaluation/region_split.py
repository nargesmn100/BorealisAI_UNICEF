"""
Region-Based Train/Test Splitting and Generalisation Evaluation
===============================================================

Why region-based splitting?
----------------------------
Nearby grid cells share spatial structure — land cover, road networks, building
density all vary smoothly across space. A random cell-level split will include
training cells that are immediate neighbours of test cells, making generalisation
look artificially good because the model has already seen the spatial context.

The problem statement requires two levels of generalisation:

  1. Across-region: hold out an entire zone (e.g. KMA).
     Train on the remaining zones only. Predict on the held-out zone.
     → Tests: can the model apply learned feature→poverty relationships
       to an area it has never seen?

  2. Within-region: hold out a spatial subarea inside a training zone.
     → Tests: can the model generalise locally within a familiar region?

Jamaica has 3 zones (Urban, Rural, KMA). With such few zones, we do
leave-one-zone-out (LOSO) cross-validation:
  - Iteration 1: train on Urban + Rural, test on KMA
  - Iteration 2: train on Urban + KMA,  test on Rural
  - Iteration 3: train on Rural + KMA,  test on Urban

At each iteration we:
  1. Train the model using only cells from the 2 training zones
     (with their 2 zone-level supervision signals)
  2. Predict on all cells of the held-out zone
  3. Aggregate predictions (population-weighted) and compare to the
     official target for the held-out zone
  4. Report the zone-level generalisation error

This is the "across-region" test. The within-region test is approximated
by leaving out a random 20% of cells from each training zone and evaluating
the prediction distribution (we can't evaluate cell-level accuracy — no
ground truth — but we can check spatial coherence via -RWI correlation).

Outputs
-------
  loso_summary : DataFrame with one row per held-out zone
    - zone_held_out
    - target (official poverty rate)
    - predicted_raw (population-weighted aggregate, NO reconciliation)
    - abs_error_raw (how far from target WITHOUT any reconciliation)
    - spearman_vs_neg_rwi (within held-out zone, correlation with -RWI)
    - pearson_vs_neg_rwi
"""

import logging
from typing import Optional, Type

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core LOSO runner
# ---------------------------------------------------------------------------

def leave_one_zone_out(
    model_cls,
    model_kwargs: dict,
    df: pd.DataFrame,
    feature_cols: list,
    zone_col: str,
    pop_col: str,
    target_col: str,
    rwi_col: str = "rwi",
) -> pd.DataFrame:
    """
    Leave-one-zone-out cross-validation for weakly supervised spatial models.

    For each zone in the dataset:
      - Train the model on cells from all OTHER zones
      - The supervision signal is the zone-level targets for training zones
      - Predict on cells of the held-out zone
      - Report zone-level aggregate error (tests across-region generalisation)

    Parameters
    ----------
    model_cls : class
        WeaklySupervisedLinear or WeaklySupervisedMLP (from weakly_supervised_model.py)
    model_kwargs : dict
        Kwargs passed to model_cls constructor.
    df : pd.DataFrame
        Modeling table (in_modeling_sample, features, targets, zone labels).
    feature_cols : list
        Feature columns to use.
    zone_col : str
        Column naming the zone (e.g. "subregion").
    pop_col : str
        Population column for weighted aggregation.
    target_col : str
        Official zone-level target column (e.g. "moderate_prevalence").
    rwi_col : str
        Column with Relative Wealth Index (for spatial coherence check).

    Returns
    -------
    pd.DataFrame
        LOSO summary with one row per held-out zone.
    """
    # Filter to modeling sample with complete features
    model_mask = df["in_modeling_sample"].fillna(False)
    feature_mask = df[feature_cols].notna().all(axis=1)
    valid_mask = model_mask & feature_mask & df[zone_col].notna() & (df[zone_col] != "Unknown")

    df_valid = df.loc[valid_mask].copy().reset_index(drop=True)

    zones = sorted(df_valid[zone_col].unique())
    n_zones = len(zones)

    logger.info(
        "LOSO CV: %d zones, %d valid cells. "
        "Running %d leave-one-out iterations...",
        n_zones, len(df_valid), n_zones,
    )

    if n_zones < 2:
        logger.warning("Fewer than 2 zones — LOSO CV not possible.")
        return pd.DataFrame()

    rows = []

    for held_out_zone in zones:
        train_zones = [z for z in zones if z != held_out_zone]
        train_mask = df_valid[zone_col].isin(train_zones)
        test_mask = df_valid[zone_col] == held_out_zone

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        logger.info(
            "LOSO fold: held_out='%s' | train_zones=%s | "
            "n_train=%d cells | n_test=%d cells",
            held_out_zone, train_zones, n_train, n_test,
        )

        if n_train < 5 or n_test < 1:
            logger.warning("Skipping fold — insufficient cells.")
            continue

        # Build training inputs
        X_train = df_valid.loc[train_mask, feature_cols]
        zone_labels_train = df_valid.loc[train_mask, zone_col].values
        pop_train = df_valid.loc[train_mask, pop_col].values.astype(float)

        # Zone targets for training zones only
        zone_targets_train = (
            df_valid.loc[train_mask]
            .groupby(zone_col)[target_col]
            .first()
            .to_dict()
        )

        logger.info("  Training zone targets: %s", zone_targets_train)

        # Official target for the held-out zone (what we want to match)
        held_out_target = df_valid.loc[test_mask, target_col].iloc[0]

        # Fit model on training zones
        model = model_cls(**model_kwargs)
        try:
            model.fit(
                X_train, zone_labels_train, pop_train,
                zone_targets_train, feature_names=feature_cols,
            )
        except Exception as e:
            logger.error("LOSO fold '%s': model fit failed: %s", held_out_zone, e)
            continue

        # Predict on held-out zone cells
        X_test = df_valid.loc[test_mask, feature_cols]
        pop_test = df_valid.loc[test_mask, pop_col].values.astype(float)
        pop_test = np.where(np.isnan(pop_test) | (pop_test < 0), 0.0, pop_test)
        rwi_test = df_valid.loc[test_mask, rwi_col].values.astype(float)

        try:
            y_hat_test = model.predict(X_test)
        except Exception as e:
            logger.error("LOSO fold '%s': prediction failed: %s", held_out_zone, e)
            continue

        # Zone-level aggregate prediction (population-weighted)
        if pop_test.sum() > 0:
            y_zone_pred = np.average(y_hat_test, weights=pop_test)
        else:
            y_zone_pred = y_hat_test.mean()

        abs_error = abs(y_zone_pred - held_out_target)

        # Spatial coherence: Spearman and Pearson correlation with -RWI
        # (higher deprivation should correlate with lower wealth)
        valid_rwi = ~np.isnan(rwi_test) & ~np.isnan(y_hat_test)
        if valid_rwi.sum() >= 3:
            spearman_r, _ = stats.spearmanr(-rwi_test[valid_rwi], y_hat_test[valid_rwi])
            pearson_r, _ = stats.pearsonr(-rwi_test[valid_rwi], y_hat_test[valid_rwi])
        else:
            spearman_r = pearson_r = float("nan")

        row = {
            "zone_held_out": held_out_zone,
            "train_zones": str(train_zones),
            "n_train_cells": int(n_train),
            "n_test_cells": int(n_test),
            "official_target": float(held_out_target),
            "predicted_aggregate_raw": float(y_zone_pred),
            "abs_error_pp": float(abs_error),
            "spearman_vs_neg_rwi": float(spearman_r),
            "pearson_vs_neg_rwi": float(pearson_r),
        }
        rows.append(row)

        logger.info(
            "  Held-out '%s': target=%.3f | pred=%.3f | "
            "abs_error=%.4f pp | spearman(−RWI)=%.3f",
            held_out_zone, held_out_target, y_zone_pred,
            abs_error, spearman_r,
        )

    summary_df = pd.DataFrame(rows)

    if len(summary_df) > 0:
        logger.info(
            "\nLOSO CV Summary (%s):\n"
            "  Mean abs_error_pp: %.4f\n"
            "  Mean spearman_vs_neg_rwi: %.3f",
            target_col,
            summary_df["abs_error_pp"].mean(),
            summary_df["spearman_vs_neg_rwi"].mean(),
        )

    return summary_df


# ---------------------------------------------------------------------------
# Within-zone subregion holdout
# ---------------------------------------------------------------------------

def within_zone_spatial_holdout(
    model_cls,
    model_kwargs: dict,
    df: pd.DataFrame,
    feature_cols: list,
    zone_col: str,
    pop_col: str,
    target_col: str,
    rwi_col: str = "rwi",
    holdout_fraction: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Within-zone spatial holdout: randomly exclude 20% of cells from each
    training zone and evaluate spatial coherence (vs -RWI) on those cells.

    Because we have no fine-scale ground truth, we cannot compute a cell-level
    accuracy metric. Instead we:
      1. Hold out 20% of cells from each zone (still use full zone targets)
      2. Train on the 80% remaining cells
      3. Predict on the held-out 20%
      4. Report Spearman correlation of predictions with -RWI

    This tests whether the model's spatial ordering is consistent even
    on cells it never saw during training.

    Parameters
    ----------
    holdout_fraction : fraction of cells to hold out per zone (default 0.20)

    Returns
    -------
    pd.DataFrame with one row per zone showing within-zone spatial coherence.
    """
    model_mask = df["in_modeling_sample"].fillna(False)
    feature_mask = df[feature_cols].notna().all(axis=1)
    valid_mask = model_mask & feature_mask & df[zone_col].notna() & (df[zone_col] != "Unknown")

    df_valid = df.loc[valid_mask].copy().reset_index(drop=True)
    zones = sorted(df_valid[zone_col].unique())
    rng = np.random.default_rng(random_state)

    logger.info(
        "Within-zone spatial holdout (holdout_fraction=%.2f): "
        "%d zones, %d valid cells.",
        holdout_fraction, len(zones), len(df_valid),
    )

    # Assign holdout membership per zone
    holdout_idx = []
    for zone in zones:
        z_idx = df_valid.index[df_valid[zone_col] == zone].tolist()
        n_hold = max(1, int(len(z_idx) * holdout_fraction))
        held = rng.choice(z_idx, size=n_hold, replace=False)
        holdout_idx.extend(held.tolist())

    holdout_set = set(holdout_idx)
    train_idx = [i for i in df_valid.index if i not in holdout_set]

    df_train = df_valid.loc[train_idx]
    df_hold = df_valid.loc[holdout_idx]

    X_train = df_train[feature_cols]
    zone_labels_train = df_train[zone_col].values
    pop_train = df_train[pop_col].values.astype(float)

    # Zone targets (from full zone, not just training cells)
    zone_targets = (
        df_valid.groupby(zone_col)[target_col].first().to_dict()
    )

    model = model_cls(**model_kwargs)
    try:
        model.fit(
            X_train, zone_labels_train, pop_train,
            zone_targets, feature_names=feature_cols,
        )
    except Exception as e:
        logger.error("Within-zone holdout: model fit failed: %s", e)
        return pd.DataFrame()

    rows = []
    for zone in zones:
        zone_hold = df_hold[df_hold[zone_col] == zone]
        if len(zone_hold) < 3:
            continue

        X_test = zone_hold[feature_cols]
        rwi_test = zone_hold[rwi_col].values.astype(float)
        y_hat_test = model.predict(X_test)

        valid = ~np.isnan(rwi_test) & ~np.isnan(y_hat_test)
        if valid.sum() >= 3:
            spearman_r, _ = stats.spearmanr(-rwi_test[valid], y_hat_test[valid])
            pearson_r, _ = stats.pearsonr(-rwi_test[valid], y_hat_test[valid])
        else:
            spearman_r = pearson_r = float("nan")

        rows.append({
            "zone": zone,
            "n_train_cells": int((df_train[zone_col] == zone).sum()),
            "n_holdout_cells": int(len(zone_hold)),
            "spearman_vs_neg_rwi_holdout": float(spearman_r),
            "pearson_vs_neg_rwi_holdout": float(pearson_r),
        })

        logger.info(
            "  Within-zone holdout '%s': n_holdout=%d | "
            "spearman(pred, −RWI)=%.3f | pearson=%.3f",
            zone, len(zone_hold), spearman_r, pearson_r,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full evaluation runner for region splits
# ---------------------------------------------------------------------------

def run_region_split_evaluation(
    df: pd.DataFrame,
    cfg: dict,
) -> dict:
    """
    Run the full region-split evaluation for both WS models.

    Returns a dict with:
      {
        "ws_linear_loso": DataFrame (LOSO results for linear model),
        "ws_mlp_loso":    DataFrame (LOSO results for MLP),
        "ws_linear_within_zone": DataFrame,
        "ws_mlp_within_zone":    DataFrame,
      }
    """
    from src.models.weakly_supervised_model import (
        WeaklySupervisedLinear, WeaklySupervisedMLP
    )

    feature_cols = cfg["modeling"]["features"]
    zone_col = cfg["modeling"]["admin_zone_col"]
    pop_col = "population"
    target_moderate = "moderate_prevalence"
    ws_cfg = cfg["modeling"].get("weakly_supervised", {})

    linear_cfg = ws_cfg.get("linear", {})
    mlp_cfg = ws_cfg.get("mlp", {})

    linear_kwargs = {
        "l2_reg": linear_cfg.get("l2_reg", 1.0),
        "random_state": linear_cfg.get("random_state", 42),
    }
    mlp_kwargs = {
        "hidden_size": mlp_cfg.get("hidden_size", 16),
        "l2_reg": mlp_cfg.get("l2_reg", 0.01),
        "max_iter": mlp_cfg.get("max_iter", 2000),
        "random_state": mlp_cfg.get("random_state", 42),
    }

    results = {}

    # --- Linear LOSO ---
    logger.info("\n=== LOSO CV: WeaklySupervisedLinear ===")
    results["ws_linear_loso"] = leave_one_zone_out(
        WeaklySupervisedLinear, linear_kwargs,
        df, feature_cols, zone_col, pop_col, target_moderate,
    )

    # --- MLP LOSO ---
    logger.info("\n=== LOSO CV: WeaklySupervisedMLP ===")
    results["ws_mlp_loso"] = leave_one_zone_out(
        WeaklySupervisedMLP, mlp_kwargs,
        df, feature_cols, zone_col, pop_col, target_moderate,
    )

    # --- Linear within-zone ---
    logger.info("\n=== Within-zone holdout: WeaklySupervisedLinear ===")
    results["ws_linear_within_zone"] = within_zone_spatial_holdout(
        WeaklySupervisedLinear, linear_kwargs,
        df, feature_cols, zone_col, pop_col, target_moderate,
    )

    # --- MLP within-zone ---
    logger.info("\n=== Within-zone holdout: WeaklySupervisedMLP ===")
    results["ws_mlp_within_zone"] = within_zone_spatial_holdout(
        WeaklySupervisedMLP, mlp_kwargs,
        df, feature_cols, zone_col, pop_col, target_moderate,
    )

    # Summary log
    logger.info("\n=== Region Split Evaluation Summary ===")
    for key, df_res in results.items():
        if isinstance(df_res, pd.DataFrame) and len(df_res) > 0:
            logger.info("\n%s:\n%s", key, df_res.to_string(index=False))

    return results
