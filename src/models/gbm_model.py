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

Interpretability strategy
--------------------------
- Feature importances (gain-based) are logged
- SHAP values are computed if shap is available
- Partial dependence plots can be produced via the evaluate pipeline

Output columns:
  gbm_raw_score           : raw GBM prediction score
  gbm_moderate            : reconciled moderate prevalence prediction
  gbm_severe              : reconciled severe prevalence prediction
  gbm_moderate_upper/lower: bootstrap uncertainty bounds
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.reconciliation.admin_reconcile import reconcile_predictions, reconcile_uncertainty_bounds
from src.utils.config_loader import get_available_features

logger = logging.getLogger(__name__)


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


def _try_shap(
    model, X_pred: np.ndarray, feature_names: list, output_dir: str | None = None,
) -> Optional[np.ndarray]:
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

        # Save SHAP values to disk
        if output_dir is not None:
            import os
            os.makedirs(output_dir, exist_ok=True)

            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            shap_path = os.path.join(output_dir, "shap_values.csv")
            shap_df.to_csv(shap_path, index=False)
            logger.info("SHAP values saved to: %s", shap_path)

            summary_df = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            summary_path = os.path.join(output_dir, "shap_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info("SHAP summary saved to: %s", summary_path)

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
    feature_cols = get_available_features(cfg, df)
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
    eval_dir = cfg.get("paths", {}).get("eval_dir")
    _try_shap(model, X_pred, feature_cols, output_dir=eval_dir)

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
    # Reconcile depth metrics (same raw score, different targets)
    # ------------------------------------------------------------------
    if "moderate_depth" in df.columns:
        logger.info("Reconciling GBM predictions to depth targets...")
        df = reconcile_predictions(
            df, raw_score_col="gbm_raw_score", target_col="moderate_depth",
            zone_col=zone_col, population_col=pop_col,
            output_col="gbm_moderate_depth", strategy="population_weighted",
        )
        df = reconcile_predictions(
            df, raw_score_col="gbm_raw_score", target_col="severe_depth",
            zone_col=zone_col, population_col=pop_col,
            output_col="gbm_severe_depth", strategy="population_weighted",
        )

    # ------------------------------------------------------------------
    # Bootstrap uncertainty
    # ------------------------------------------------------------------
    n_bootstrap = uncertainty_cfg.get("n_bootstrap", 50)
    rs = uncertainty_cfg.get("random_state", 42)
    rng = np.random.default_rng(rs)

    logger.info("Running bootstrap uncertainty (%d samples)...", n_bootstrap)
    boot_preds = np.zeros((n_bootstrap, len(X_pred)))

    for b in range(n_bootstrap):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        m = _get_gbm_model(cfg)
        m.fit(X_train[idx], y_train[idx])
        boot_preds[b] = m.predict(X_pred)

    df["gbm_moderate_lower"] = np.nan
    df["gbm_moderate_upper"] = np.nan
    df.loc[pred_mask, "gbm_moderate_lower"] = np.percentile(boot_preds, 5, axis=0)
    df.loc[pred_mask, "gbm_moderate_upper"] = np.percentile(boot_preds, 95, axis=0)

    # Propagate uncertainty bounds through reconciliation
    df = reconcile_uncertainty_bounds(
        df,
        lower_col="gbm_moderate_lower",
        upper_col="gbm_moderate_upper",
        raw_col="gbm_raw_score",
        reconciled_col="gbm_moderate",
        zone_col=zone_col,
        population_col=pop_col,
    )

    logger.info("GBM model complete.")
    return df, model, fi_df
