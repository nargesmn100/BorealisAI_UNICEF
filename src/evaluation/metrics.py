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
from scipy.spatial import cKDTree

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
# Non-circular evaluation metrics
# ---------------------------------------------------------------------------

def spatial_smoothness(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    k: int = 8,
) -> float:
    """
    Measure spatial autocorrelation via KNN neighbor correlation.

    For each cell, computes the mean value of its k nearest neighbors, then
    returns the Pearson correlation between cell values and neighbor means.
    Higher values indicate spatially smoother (more coherent) predictions.

    Parameters
    ----------
    lons, lats : np.ndarray
        Coordinates.
    values : np.ndarray
        Prediction values.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    float
        Pearson r between values and spatial lag (neighbor mean).
    """
    valid = ~np.isnan(values)
    if valid.sum() < k + 1:
        return np.nan

    coords = np.column_stack([lons[valid], lats[valid]])
    vals = values[valid]
    tree = cKDTree(coords)

    # k+1 because the first neighbor is the point itself
    distances, indices = tree.query(coords, k=k + 1)
    # Exclude self (index 0), take mean of neighbors
    neighbor_means = np.mean(vals[indices[:, 1:]], axis=1)

    r, _ = stats.pearsonr(vals, neighbor_means)
    return float(r)


def multi_proxy_agreement(
    df: pd.DataFrame,
    pred_col: str,
) -> dict:
    """
    Compute Spearman correlations between predictions and non-RWI proxies.

    This provides non-circular evaluation metrics (unlike RWI correlation which
    is circular for the RWI baseline).

    Expected correlations for deprivation predictions:
    - travel_time_cities: positive (remote areas → more deprived)
    - smod_class: negative (urban areas → less deprived)
    - population: negative (populated areas → less deprived, generally)

    Parameters
    ----------
    df : pd.DataFrame
    pred_col : str

    Returns
    -------
    dict
        Keys: proxy_r_travel_time, proxy_r_smod, proxy_r_population
    """
    pred = df[pred_col].values.astype(float)
    result = {}

    proxy_cols = {
        "travel_time_cities": "proxy_r_travel_time",
        "smod_class": "proxy_r_smod",
        "population": "proxy_r_population",
    }

    for col, key in proxy_cols.items():
        if col not in df.columns:
            result[key] = np.nan
            continue
        proxy = df[col].values.astype(float)
        mask = ~(np.isnan(pred) | np.isnan(proxy))
        if mask.sum() < 3:
            result[key] = np.nan
            continue
        r, _ = stats.spearmanr(pred[mask], proxy[mask])
        result[key] = float(r)

    return result


def within_zone_variation(
    df: pd.DataFrame,
    pred_col: str,
    zone_col: str = "subregion",
) -> dict:
    """
    Compute coefficient of variation (CV) of predictions within each zone.

    Uniform allocation gives CV=0. Methods with meaningful spatial variation
    should have higher CV, indicating they differentiate within zones.

    Parameters
    ----------
    df : pd.DataFrame
    pred_col : str
    zone_col : str

    Returns
    -------
    dict
        Keys: cv_{zone_name} and cv_overall
    """
    result = {}
    valid = df[pred_col].notna() & (df[zone_col] != "Unknown")

    for zone in sorted(df.loc[valid, zone_col].unique()):
        zmask = valid & (df[zone_col] == zone)
        vals = df.loc[zmask, pred_col].values
        if len(vals) < 2 or np.mean(vals) == 0:
            cv = 0.0
        else:
            cv = float(np.std(vals) / np.abs(np.mean(vals)))
        # Sanitize zone name for column key
        zone_key = zone.lower().replace(" ", "_").replace("(", "").replace(")", "")
        result[f"cv_{zone_key}"] = cv

    # Overall CV
    all_vals = df.loc[valid, pred_col].values
    if len(all_vals) > 1 and np.mean(all_vals) != 0:
        result["cv_overall"] = float(np.std(all_vals) / np.abs(np.mean(all_vals)))
    else:
        result["cv_overall"] = 0.0

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
        # Spatial smoothness
        # ----------------------------------------------------------------
        smoothness = spatial_smoothness(
            df["longitude"].values, df["latitude"].values, pred_vals,
        )

        # ----------------------------------------------------------------
        # Multi-proxy agreement (non-circular)
        # ----------------------------------------------------------------
        proxy_results = multi_proxy_agreement(df, pred_col)

        # ----------------------------------------------------------------
        # Within-zone variation
        # ----------------------------------------------------------------
        cv_results = within_zone_variation(df, pred_col, zone_col)

        # ----------------------------------------------------------------
        # Compile results
        # ----------------------------------------------------------------
        method_results = {
            "admin_mae_mean_pp": mean_admin_mae,
            "pearson_r_vs_neg_rwi": pearson_pred_rwi,
            "spearman_r_vs_neg_rwi": spearman_pred_rwi,
            "spatial_smoothness": smoothness,
            **proxy_results,
            **cv_results,
            **top_k_results,
            **{f"ci_{k}": v for k, v in ci_stats.items()},
            "admin_detail": admin_err_df,
        }
        results[method] = method_results

        logger.info(
            "Method %-20s | admin_mae=%.4f pp | pearson(pred,−RWI)=%.3f | spearman=%.3f | smoothness=%.3f",
            method, mean_admin_mae, pearson_pred_rwi, spearman_pred_rwi, smoothness,
        )

    # ----------------------------------------------------------------
    # Cross-method comparison summary
    # ----------------------------------------------------------------
    logger.info("\n=== Evaluation Summary ===")
    logger.info(
        "%-20s | %12s | %12s | %12s | %10s | %10s | %10s",
        "Method", "admin_MAE(pp)", "pearson(−RWI)", "spearman(−RWI)",
        "smoothness", "proxy_r_tt", "cv_overall",
    )
    logger.info("-" * 100)
    for method, res in results.items():
        logger.info(
            "%-20s | %12.4f | %12.3f | %12.3f | %10.3f | %10.3f | %10.3f",
            method,
            res.get("admin_mae_mean_pp", np.nan),
            res.get("pearson_r_vs_neg_rwi", np.nan),
            res.get("spearman_r_vs_neg_rwi", np.nan),
            res.get("spatial_smoothness", np.nan),
            res.get("proxy_r_travel_time", np.nan),
            res.get("cv_overall", np.nan),
        )

    # ----------------------------------------------------------------
    # Depth metric evaluation
    # ----------------------------------------------------------------
    depth_methods = {}
    for col in df.columns:
        if col.endswith("_moderate_depth") and not col.endswith("_lower") and not col.endswith("_upper"):
            method = col.replace("_moderate_depth", "")
            depth_methods[method] = col

    if depth_methods and "moderate_depth" in df.columns:
        logger.info("\n=== Depth Metric Evaluation ===")
        for method, pred_col in depth_methods.items():
            if pred_col not in df.columns:
                continue
            pred_vals = df[pred_col].values.astype(float)
            rwi_vals = df["rwi"].values.astype(float)

            admin_err_df = admin_consistency_error(
                df, pred_col, "moderate_depth", zone_col, pop_col,
            )
            mean_admin_mae_d = admin_err_df["abs_error"].mean() if len(admin_err_df) > 0 else np.nan
            pearson_pred_rwi_d = pearson_r(-rwi_vals, pred_vals)
            spearman_pred_rwi_d = spearman_r(-rwi_vals, pred_vals)

            depth_key = f"{method}_depth"
            results[depth_key] = {
                "admin_mae_mean_pp": mean_admin_mae_d,
                "pearson_r_vs_neg_rwi": pearson_pred_rwi_d,
                "spearman_r_vs_neg_rwi": spearman_pred_rwi_d,
                "admin_detail": admin_err_df,
                "metric_type": "depth",
            }
            logger.info(
                "Method %-20s | admin_mae=%.4f | pearson=%.3f | spearman=%.3f",
                depth_key, mean_admin_mae_d, pearson_pred_rwi_d, spearman_pred_rwi_d,
            )

    logger.info(
        "\nIMPORTANT CAVEAT: Metrics compare predictions against RWI as a proxy "
        "for spatial truth. Correlations with RWI measure proxy agreement, not "
        "true deprivation reconstruction accuracy. This limitation should be "
        "stated explicitly in any report."
    )

    return results


def paired_significance_tests(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Test whether ML methods significantly differ from the RWI baseline
    using Wilcoxon signed-rank test on per-cell absolute error differences,
    and bootstrap CIs for difference in Spearman r.

    Parameters
    ----------
    df : pd.DataFrame
    cfg : dict

    Returns
    -------
    pd.DataFrame
        Columns: method_pair, test_statistic, p_value, direction, ci_lower, ci_upper
    """
    if "rwi_moderate" not in df.columns:
        logger.warning("RWI baseline not found. Skipping significance tests.")
        return pd.DataFrame()

    rwi_vals = df["rwi"].values.astype(float)
    neg_rwi = -rwi_vals
    rwi_preds = df["rwi_moderate"].values.astype(float)

    ml_methods = []
    for col in df.columns:
        if col.endswith("_moderate") and col not in [
            "rwi_moderate", "uniform_moderate",
        ] and not col.endswith("_lower") and not col.endswith("_upper") and not col.endswith("_depth"):
            ml_methods.append(col)

    rows = []
    valid_base = ~(np.isnan(rwi_preds) | np.isnan(neg_rwi))

    for ml_col in ml_methods:
        ml_preds = df[ml_col].values.astype(float)
        valid = valid_base & ~np.isnan(ml_preds)
        if valid.sum() < 10:
            continue

        # Per-cell |error| vs -RWI proxy
        rwi_errors = np.abs(rwi_preds[valid] - neg_rwi[valid])
        ml_errors = np.abs(ml_preds[valid] - neg_rwi[valid])
        diff = rwi_errors - ml_errors  # positive = ML is better

        try:
            stat, p_val = stats.wilcoxon(diff, alternative="two-sided")
        except ValueError:
            stat, p_val = np.nan, np.nan

        direction = "ml_better" if np.median(diff) > 0 else "rwi_better"

        # Bootstrap CI for Spearman r difference
        rng = np.random.default_rng(42)
        n = valid.sum()
        rwi_sp = spearman_r(neg_rwi[valid], rwi_preds[valid])
        ml_sp = spearman_r(neg_rwi[valid], ml_preds[valid])

        boot_diffs = []
        for _ in range(1000):
            idx = rng.integers(0, n, size=n)
            sp_rwi_b = spearman_r(neg_rwi[valid][idx], rwi_preds[valid][idx])
            sp_ml_b = spearman_r(neg_rwi[valid][idx], ml_preds[valid][idx])
            boot_diffs.append(sp_ml_b - sp_rwi_b)

        boot_diffs = np.array(boot_diffs)
        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))

        method_name = ml_col.replace("_moderate", "")
        rows.append({
            "method_pair": f"{method_name}_vs_rwi",
            "test_statistic": float(stat) if not np.isnan(stat) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
            "direction": direction,
            "spearman_diff": ml_sp - rwi_sp,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        logger.info("Significance tests:\n%s", result.to_string(index=False))
    return result


def rwi_uncertainty_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse the relationship between RWI posterior uncertainty (rwi_error)
    and prediction CI widths.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary statistics by zone and overall.
    """
    if "rwi_error" not in df.columns:
        logger.warning("rwi_error column not found. Skipping RWI uncertainty analysis.")
        return pd.DataFrame()

    valid = df["in_modeling_sample"].fillna(False) & df["rwi_error"].notna()
    rows = []

    # Per-zone stats
    for zone in sorted(df.loc[valid, "subregion"].unique()):
        zmask = valid & (df["subregion"] == zone)
        rwi_err = df.loc[zmask, "rwi_error"].values
        row = {
            "zone": zone,
            "n_cells": int(zmask.sum()),
            "mean_rwi_error": float(rwi_err.mean()),
            "median_rwi_error": float(np.median(rwi_err)),
            "high_uncertainty_count": int((rwi_err > 0.6).sum()),
        }

        # CI width correlation if available
        for prefix in ["ridge", "gam", "gbm"]:
            lower_col = f"{prefix}_moderate_lower"
            upper_col = f"{prefix}_moderate_upper"
            if lower_col in df.columns and upper_col in df.columns:
                ci_width = (df.loc[zmask, upper_col] - df.loc[zmask, lower_col]).values
                ci_valid = ~np.isnan(ci_width)
                if ci_valid.sum() > 3:
                    corr, _ = stats.pearsonr(rwi_err[ci_valid], ci_width[ci_valid])
                    row[f"{prefix}_ci_width_corr"] = float(corr)

        rows.append(row)

    # Overall
    rwi_err_all = df.loc[valid, "rwi_error"].values
    overall = {
        "zone": "ALL",
        "n_cells": int(valid.sum()),
        "mean_rwi_error": float(rwi_err_all.mean()),
        "median_rwi_error": float(np.median(rwi_err_all)),
        "high_uncertainty_count": int((rwi_err_all > 0.6).sum()),
    }
    rows.append(overall)

    result = pd.DataFrame(rows)
    logger.info("RWI uncertainty analysis:\n%s", result.to_string(index=False))
    return result


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
