"""
Evaluation Metrics Module

Compares model predictions against baselines and available reference values.

Evaluation context
------------------
Because official poverty data is only available at 3 subnational zones
(Urban / Rural / KMA) for Jamaica, the primary evaluation compares:

1. How well each method assigns *relative* deprivation rankings within zones
   (based on proxy agreement: correlation of predictions with RWI rank)

2. Admin consistency: do reconciled predictions exactly match zone targets?

3. Cross-zone evaluation: how well does each method distinguish between zones?
   This tests whether methods correctly rank Urban < Rural < KMA in deprivation.

4. Uncertainty calibration: are confidence intervals appropriately wide?

Since we do NOT have fine-resolution ground truth, all accuracy metrics
should be interpreted as proxy-agreement metrics, not ground truth error.

This limitation is explicitly logged and documented.

Metrics computed
----------------
  mae            : mean absolute error (vs zone-level reference)
  rmse           : root mean squared error
  pearson_r      : Pearson correlation
  spearman_r     : Spearman rank correlation
  top_k_overlap  : proportion of top-K cells that overlap across methods
  admin_error    : zone-level mean absolute error from official targets
  ci_width       : mean 90% confidence interval width (for uncertainty models)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pearsonr(y_true[mask], y_pred[mask])
    return float(r)


def spearman_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.spearmanr(y_true[mask], y_pred[mask])
    return float(r)


def top_k_overlap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    k: int,
) -> float:
    """
    Fraction of top-K cells (by scores_a) that also appear in top-K of scores_b.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
    k : int

    Returns
    -------
    float in [0, 1]
    """
    mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
    if mask.sum() < k:
        logger.warning("top_k_overlap: fewer valid cells (%d) than k=%d.", mask.sum(), k)
        k = mask.sum()

    idx_a = np.argsort(scores_a[mask])[-k:]  # top-k indices in valid array
    idx_b = np.argsort(scores_b[mask])[-k:]
    overlap = len(set(idx_a) & set(idx_b)) / k
    return float(overlap)


def admin_consistency_error(
    df: pd.DataFrame,
    pred_col: str,
    target_col: str,
    zone_col: str,
    population_col: str,
) -> pd.DataFrame:
    """
    Compute zone-level mean absolute error between reconciled predictions
    and official targets (population-weighted mean vs target).

    Parameters
    ----------
    df : pd.DataFrame
    pred_col : str
    target_col : str
    zone_col : str
    population_col : str

    Returns
    -------
    pd.DataFrame
        One row per zone with columns: zone, target, achieved, abs_error.
    """
    valid = df[pred_col].notna() & df[target_col].notna()
    zones = df.loc[valid, zone_col].unique()
    rows = []

    for zone in sorted(zones):
        zmask = valid & (df[zone_col] == zone)
        target = df.loc[zmask, target_col].iloc[0]
        pop = df.loc[zmask, population_col].values.astype(float)
        pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)
        preds = df.loc[zmask, pred_col].values

        achieved = np.average(preds, weights=pop) if pop.sum() > 0 else preds.mean()
        rows.append({
            "zone": zone,
            "target": target,
            "achieved": achieved,
            "abs_error": abs(achieved - target),
        })

    return pd.DataFrame(rows)


def ci_coverage_width(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute confidence interval statistics.

    Parameters
    ----------
    lower, upper : np.ndarray
        Lower and upper bounds of prediction interval.
    y_true : np.ndarray or None
        If provided, compute empirical coverage.

    Returns
    -------
    dict with keys: mean_width, median_width, coverage (if y_true given)
    """
    mask = ~(np.isnan(lower) | np.isnan(upper))
    widths = (upper - lower)[mask]
    result = {
        "mean_width": float(widths.mean()) if len(widths) > 0 else np.nan,
        "median_width": float(np.median(widths)) if len(widths) > 0 else np.nan,
    }
    if y_true is not None:
        valid = mask & ~np.isnan(y_true)
        if valid.sum() > 0:
            in_interval = (y_true[valid] >= lower[valid]) & (y_true[valid] <= upper[valid])
            result["coverage"] = float(in_interval.mean())
    return result


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def evaluate_all(
    df: pd.DataFrame,
    cfg: dict,
) -> dict:
    """
    Run the full comparative evaluation across baselines and models.

    Computes metrics for each prediction column found in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling table with predictions from all methods.
    cfg : dict
        Loaded config.

    Returns
    -------
    dict
        Nested dict: {method_name: {metric_name: value}}
    """
    zone_col = cfg["modeling"]["admin_zone_col"]
    pop_col = "population"
    # Use the renamed column names from the modeling table (not raw Excel names)
    target_moderate = "moderate_prevalence"
    target_severe = "severe_prevalence"
    priority_k_list = cfg["evaluation"]["priority_k"]

    # Map output column names to method labels
    method_cols = {}
    for col in df.columns:
        if col.endswith("_moderate") and not col.endswith("_lower") and not col.endswith("_upper"):
            method = col.replace("_moderate", "")
            method_cols[method] = col

    logger.info("Evaluating methods: %s", list(method_cols.keys()))

    results = {}

    # Reference: zone-level targets as point values for all cells
    target_vals = df[target_moderate].values.astype(float)

    for method, pred_col in method_cols.items():
        if pred_col not in df.columns:
            logger.warning("Prediction column '%s' not found. Skipping.", pred_col)
            continue

        pred_vals = df[pred_col].values.astype(float)

        # ----------------------------------------------------------------
        # Admin consistency
        # ----------------------------------------------------------------
        admin_err_df = admin_consistency_error(
            df, pred_col, target_moderate, zone_col, pop_col
        )
        mean_admin_mae = admin_err_df["abs_error"].mean() if len(admin_err_df) > 0 else np.nan

        # ----------------------------------------------------------------
        # Within-zone spatial correlation with RWI
        # (proxy for spatial accuracy — assumes RWI is a rough ground truth)
        # ----------------------------------------------------------------
        # Overall
        rwi_vals = df["rwi"].values.astype(float)
        # For deprivation, higher RWI = wealthier = should be less deprived
        # So we expect negative Pearson r between pred and rwi
        pearson_pred_rwi = pearson_r(-rwi_vals, pred_vals)
        spearman_pred_rwi = spearman_r(-rwi_vals, pred_vals)

        # ----------------------------------------------------------------
        # Top-K overlap with RWI baseline (if available)
        # ----------------------------------------------------------------
        top_k_results = {}
        if "rwi_moderate" in df.columns:
            rwi_pred = df["rwi_moderate"].values.astype(float)
            for k in priority_k_list:
                top_k_results[f"top_{k}_overlap_vs_rwi"] = top_k_overlap(pred_vals, rwi_pred, k)

        # ----------------------------------------------------------------
        # Uncertainty metrics
        # ----------------------------------------------------------------
        lower_col = f"{method}_moderate_lower"
        upper_col = f"{method}_moderate_upper"
        ci_stats = {}
        if lower_col in df.columns and upper_col in df.columns:
            ci_stats = ci_coverage_width(
                df[lower_col].values.astype(float),
                df[upper_col].values.astype(float),
            )

        # ----------------------------------------------------------------
        # Compile results
        # ----------------------------------------------------------------
        method_results = {
            "admin_mae_mean_pp": mean_admin_mae,
            "pearson_r_vs_neg_rwi": pearson_pred_rwi,
            "spearman_r_vs_neg_rwi": spearman_pred_rwi,
            **top_k_results,
            **{f"ci_{k}": v for k, v in ci_stats.items()},
            "admin_detail": admin_err_df,
        }
        results[method] = method_results

        logger.info(
            "Method %-20s | admin_mae=%.4f pp | pearson(pred,−RWI)=%.3f | spearman=%.3f",
            method, mean_admin_mae, pearson_pred_rwi, spearman_pred_rwi,
        )

    # ----------------------------------------------------------------
    # Cross-method comparison summary
    # ----------------------------------------------------------------
    logger.info("\n=== Evaluation Summary ===")
    logger.info(
        "%-20s | %12s | %12s | %12s",
        "Method", "admin_MAE(pp)", "pearson(−RWI)", "spearman(−RWI)"
    )
    logger.info("-" * 65)
    for method, res in results.items():
        logger.info(
            "%-20s | %12.4f | %12.3f | %12.3f",
            method,
            res.get("admin_mae_mean_pp", np.nan),
            res.get("pearson_r_vs_neg_rwi", np.nan),
            res.get("spearman_r_vs_neg_rwi", np.nan),
        )

    logger.info(
        "\nIMPORTANT CAVEAT: Metrics compare predictions against RWI as a proxy "
        "for spatial truth. Correlations with RWI measure proxy agreement, not "
        "true deprivation reconstruction accuracy. This limitation should be "
        "stated explicitly in any report."
    )

    return results


def format_eval_report(results: dict) -> pd.DataFrame:
    """
    Format evaluation results into a summary DataFrame.

    Parameters
    ----------
    results : dict
        Output from evaluate_all().

    Returns
    -------
    pd.DataFrame
        Summary table with one row per method.
    """
    rows = []
    for method, res in results.items():
        row = {"method": method}
        for key, val in res.items():
            if key == "admin_detail":
                continue
            if isinstance(val, (int, float, np.floating, np.integer)):
                row[key] = float(val)
        rows.append(row)

    return pd.DataFrame(rows).set_index("method")
