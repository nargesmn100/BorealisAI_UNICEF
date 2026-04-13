"""
Gradient Boosted Trees Model — Nonlinear Benchmark

Uses LightGBM (with XGBoost as fallback) to fit a more flexible model
with the same spatial allocation framing as the ridge model.

Scientific framing
------------------
Same as ridge_model.py: the GBM predicts a relative deprivation score
per grid cell, and post-processing reconciliation maps those scores to
official zone-level prevalence values.

Why GBM?
--------
- Captures nonlinear relationships between features and deprivation
- Handles feature interactions (e.g. RWI × travel_time)
- Provides native feature importance estimates (gain-based and SHAP)
- Stronger nonlinear benchmark for comparing against ridge regression

Uncertainty estimation
-----------------------
Two complementary approaches are used:

1. Quantile regression (LightGBM only):
   Train separate GBM models with objective='quantile' at alpha=0.05 and
   alpha=0.95 to directly predict the 5th and 95th percentile of the
   conditional distribution.  Output: gbm_moderate_lower / gbm_moderate_upper.

2. Split-conformal prediction:
   Hold out a calibration fraction of training data, fit the mean model on
   the remainder, compute nonconformity scores (|y − ŷ|) on the calibration
   set, and derive a coverage-guaranteed symmetric interval.
   Output: gbm_moderate_conformal_lower / gbm_moderate_conformal_upper.

Output columns:
  gbm_raw_score                   : raw GBM mean prediction score
  gbm_moderate                    : reconciled moderate prevalence prediction
  gbm_severe                      : reconciled severe prevalence prediction
  gbm_moderate_lower              : quantile-regression lower bound (q=0.05)
  gbm_moderate_upper              : quantile-regression upper bound (q=0.95)
  gbm_moderate_conformal_lower    : conformal prediction lower bound
  gbm_moderate_conformal_upper    : conformal prediction upper bound
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.reconciliation.admin_reconcile import reconcile_predictions
from src.utils.conformal import SplitConformalPredictor, calibration_split

logger = logging.getLogger(__name__)


def _get_quantile_gbm_model(cfg: dict, alpha: float):
    """
    Return a LightGBM quantile regression model for the given quantile level.

    Only available with LightGBM backend (not XGBoost fallback).

    Parameters
    ----------
    cfg : dict
    alpha : float
        Quantile level, e.g. 0.05 for lower bound, 0.95 for upper bound.

    Returns
    -------
    lgb.LGBMRegressor configured with objective='quantile'.

    Raises
    ------
    ImportError if LightGBM is not installed.
    """
    import lightgbm as lgb
    gbm_cfg = cfg["modeling"]["gbm"]
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=gbm_cfg["n_estimators"],
        max_depth=gbm_cfg["max_depth"],
        learning_rate=gbm_cfg["learning_rate"],
        subsample=gbm_cfg["subsample"],
        colsample_bytree=gbm_cfg["colsample_bytree"],
        random_state=gbm_cfg["random_state"],
        n_jobs=gbm_cfg.get("n_jobs", -1),
        verbose=-1,
    )


def _get_gbm_model(cfg: dict):
    """
    Return a configured GBM model (LightGBM preferred, XGBoost fallback).

    Parameters
    ----------
    cfg : dict
        Full config dict.

    Returns
    -------
    Fitted-able sklearn-compatible GBM estimator.
    """
    gbm_cfg = cfg["modeling"]["gbm"]

    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=gbm_cfg["n_estimators"],
            max_depth=gbm_cfg["max_depth"],
            learning_rate=gbm_cfg["learning_rate"],
            subsample=gbm_cfg["subsample"],
            colsample_bytree=gbm_cfg["colsample_bytree"],
            random_state=gbm_cfg["random_state"],
            n_jobs=gbm_cfg.get("n_jobs", -1),
            verbose=-1,
        )
        logger.info("Using LightGBM backend.")
        return model
    except ImportError:
        pass

    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=gbm_cfg["n_estimators"],
            max_depth=gbm_cfg["max_depth"],
            learning_rate=gbm_cfg["learning_rate"],
            subsample=gbm_cfg["subsample"],
            colsample_bytree=gbm_cfg["colsample_bytree"],
            random_state=gbm_cfg["random_state"],
            n_jobs=gbm_cfg.get("n_jobs", -1),
            verbosity=0,
        )
        logger.info("Using XGBoost backend.")
        return model
    except ImportError:
        pass

    raise ImportError(
        "Neither LightGBM nor XGBoost is installed. "
        "Install one with: pip install lightgbm  or  pip install xgboost"
    )


def _log_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract and log feature importances."""
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info("GBM feature importances (gain):")
        for _, row in fi_df.iterrows():
            logger.info("  %-35s : %.4f", row["feature"], row["importance"])

        return fi_df
    except AttributeError:
        logger.warning("Could not extract feature importances from model.")
        return pd.DataFrame()


def _try_shap(model, X_pred: np.ndarray, feature_names: list) -> Optional[np.ndarray]:
    """Attempt to compute SHAP values; return None if shap not available."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_pred)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        logger.info("SHAP mean absolute values:")
        for name, val in sorted(zip(feature_names, mean_abs_shap),
                                key=lambda x: -x[1]):
            logger.info("  %-35s : %.4f", name, val)
        return shap_values
    except ImportError:
        logger.info("shap not installed — skipping SHAP analysis. "
                    "Install with: pip install shap")
        return None


def run(
    cfg: dict,
    df: pd.DataFrame,
) -> tuple:
    """
    Fit GBM model, generate predictions, reconcile, add uncertainty bounds.

    Parameters
    ----------
    cfg : dict
    df : pd.DataFrame
        Modeling table.

    Returns
    -------
    (pd.DataFrame, model, feature_importance_df)
    """
    feature_cols = cfg["modeling"]["features"]
    # Use the renamed column names from the modeling table (not raw Excel names)
    target_moderate = "moderate_prevalence"
    target_severe = "severe_prevalence"
    zone_col = cfg["modeling"]["admin_zone_col"]
    pop_col = "population"
    uncertainty_cfg = cfg["modeling"]["uncertainty"]

    # ------------------------------------------------------------------
    # Prepare training data
    # ------------------------------------------------------------------
    model_mask = df["in_modeling_sample"].fillna(False)
    feature_mask = df[feature_cols].notna().all(axis=1)
    train_mask = model_mask & feature_mask

    logger.info(
        "GBM training: %d cells with complete features.", train_mask.sum()
    )

    if train_mask.sum() < 10:
        raise ValueError(
            f"Insufficient training samples ({train_mask.sum()}). "
            "Check data pipeline."
        )

    X_train = df.loc[train_mask, feature_cols].values.astype(float)
    y_train = df.loc[train_mask, target_moderate].values.astype(float)

    # ------------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------------
    logger.info("Fitting GBM model...")
    model = _get_gbm_model(cfg)
    model.fit(X_train, y_train)

    fi_df = _log_feature_importance(model, feature_cols)

    # ------------------------------------------------------------------
    # Predict on all valid cells
    # ------------------------------------------------------------------
    pred_mask = feature_mask & (df["subregion"] != "Unknown")
    X_pred = df.loc[pred_mask, feature_cols].values.astype(float)

    raw_preds = model.predict(X_pred)

    df = df.copy()
    df["gbm_raw_score"] = np.nan
    df.loc[pred_mask, "gbm_raw_score"] = raw_preds

    # SHAP analysis
    _try_shap(model, X_pred, feature_cols)

    # ------------------------------------------------------------------
    # Reconcile predictions
    # ------------------------------------------------------------------
    logger.info("Reconciling GBM predictions to moderate targets...")
    df = reconcile_predictions(
        df,
        raw_score_col="gbm_raw_score",
        target_col=target_moderate,
        zone_col=zone_col,
        population_col=pop_col,
        output_col="gbm_moderate",
        strategy="population_weighted",
    )

    logger.info("Reconciling GBM predictions to severe targets...")
    df = reconcile_predictions(
        df,
        raw_score_col="gbm_raw_score",
        target_col=target_severe,
        zone_col=zone_col,
        population_col=pop_col,
        output_col="gbm_severe",
        strategy="population_weighted",
    )

    # ------------------------------------------------------------------
    # Quantile regression (LightGBM only)
    # Trains two additional models at alpha=0.05 and alpha=0.95 to
    # directly predict the 5th/95th percentiles of the conditional
    # distribution, giving asymmetric, feature-aware bounds.
    # ------------------------------------------------------------------
    df["gbm_moderate_lower"] = np.nan
    df["gbm_moderate_upper"] = np.nan

    try:
        coverage = cfg["modeling"].get("conformal", {}).get("coverage", 0.90)
        lower_alpha = (1.0 - coverage) / 2.0          # e.g. 0.05
        upper_alpha = 1.0 - lower_alpha                # e.g. 0.95

        logger.info(
            "Fitting quantile GBM models (q=%.2f, q=%.2f)...",
            lower_alpha, upper_alpha,
        )
        lower_q_model = _get_quantile_gbm_model(cfg, alpha=lower_alpha)
        upper_q_model = _get_quantile_gbm_model(cfg, alpha=upper_alpha)
        lower_q_model.fit(X_train, y_train)
        upper_q_model.fit(X_train, y_train)

        df.loc[pred_mask, "gbm_moderate_lower"] = lower_q_model.predict(X_pred)
        df.loc[pred_mask, "gbm_moderate_upper"] = upper_q_model.predict(X_pred)
        logger.info(
            "Quantile regression intervals: mean width=%.4f pp",
            (df["gbm_moderate_upper"] - df["gbm_moderate_lower"]).mean(),
        )
    except ImportError:
        logger.warning(
            "Quantile regression requires LightGBM. "
            "XGBoost fallback does not support quantile objective — "
            "gbm_moderate_lower/upper will be NaN."
        )

    # ------------------------------------------------------------------
    # Split-conformal prediction intervals
    # Uses a held-out calibration split to produce coverage-guaranteed
    # symmetric intervals around the mean model's predictions.
    # ------------------------------------------------------------------
    conformal_cfg = cfg["modeling"].get("conformal", {})
    coverage = conformal_cfg.get("coverage", 0.90)
    cal_fraction = conformal_cfg.get("cal_fraction", 0.20)
    rs = uncertainty_cfg.get("random_state", 42)

    logger.info(
        "Fitting conformal predictor (coverage=%.0f%%, cal_fraction=%.0f%%)...",
        coverage * 100, cal_fraction * 100,
    )
    X_train_fit, X_cal, y_train_fit, y_cal = calibration_split(
        X_train, y_train, cal_fraction=cal_fraction, random_state=rs
    )
    conformal_model = _get_gbm_model(cfg)
    conformal_model.fit(X_train_fit, y_train_fit)
    y_hat_cal = conformal_model.predict(X_cal)

    conformal = SplitConformalPredictor(coverage=coverage)
    conformal.calibrate(y_cal, y_hat_cal)

    # Re-use the already-predicted raw_preds from the full-data model
    conformal_lower, conformal_upper = conformal.predict_intervals(raw_preds)
    df["gbm_moderate_conformal_lower"] = np.nan
    df["gbm_moderate_conformal_upper"] = np.nan
    df.loc[pred_mask, "gbm_moderate_conformal_lower"] = conformal_lower
    df.loc[pred_mask, "gbm_moderate_conformal_upper"] = conformal_upper
    logger.info(
        "Conformal intervals: q̂=%.4f pp, interval width=%.4f pp",
        conformal.q_hat, conformal.interval_width,
    )

    logger.info("GBM model complete.")
    return df, model, fi_df
