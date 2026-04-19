"""
Leave-One-Zone-Out Cross-Validation

Evaluates how well each method generalizes when one zone/subregion is held
out from training. Works with any number of zones (3 for Jamaica, 37 for
Nigeria, etc.).

For countries with many zones (e.g. Nigeria's 37 states), this provides
a strong test of geographic generalization.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.reconciliation.admin_reconcile import reconcile_predictions
from src.utils.config_loader import get_available_features

logger = logging.getLogger(__name__)


def leave_one_zone_out(
    df: pd.DataFrame,
    cfg: dict,
    max_zones: int | None = None,
    skip_gbm: bool = False,
    skip_gam: bool = False,
    skip_wsnn: bool = False,
) -> pd.DataFrame:
    """
    Leave-one-zone-out cross-validation for all methods.

    For each zone: train on cells from all other zones, predict on the
    held-out zone, and compute the population-weighted mean prediction
    vs the official target.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling table with features and targets.
    cfg : dict
        Loaded config.
    max_zones : int or None
        If set, randomly sample this many zones for LOZO (performance guard
        for countries with many zones like Nigeria's 37 states).

    Returns
    -------
    pd.DataFrame
        One row per (zone, method) with: zone, method, target, predicted_aggregate, abs_error.
    """
    feature_cols = get_available_features(cfg, df)
    zone_col = cfg["modeling"]["admin_zone_col"]
    pop_col = "population"
    target_col = "moderate_prevalence"

    model_mask = df["in_modeling_sample"].fillna(False)
    feature_mask = df[feature_cols].notna().all(axis=1)
    valid_mask = model_mask & feature_mask

    zones = sorted(df.loc[valid_mask, zone_col].unique())
    if len(zones) < 2:
        logger.warning("LOZO requires at least 2 zones, found %d. Skipping.", len(zones))
        return pd.DataFrame()

    # Subsample zones if there are too many (performance guard)
    if max_zones is not None and len(zones) > max_zones:
        rng = np.random.RandomState(42)
        zones = sorted(rng.choice(zones, size=max_zones, replace=False))
        logger.info("Subsampled to %d zones for LOZO (max_zones=%d).", len(zones), max_zones)

    logger.info("Running leave-one-zone-out CV over %d zones: %s", len(zones), zones)
    rows = []

    for held_out_zone in zones:
        train_mask = valid_mask & (df[zone_col] != held_out_zone)
        test_mask = valid_mask & (df[zone_col] == held_out_zone)

        if train_mask.sum() < 5 or test_mask.sum() < 5:
            logger.warning("Zone '%s': insufficient data for LOZO. Skipping.", held_out_zone)
            continue

        target_value = df.loc[test_mask, target_col].iloc[0]
        test_pop = df.loc[test_mask, pop_col].values.astype(float)
        test_pop = np.where(np.isnan(test_pop) | (test_pop < 0), 0.0, test_pop)

        X_train = df.loc[train_mask, feature_cols].values.astype(float)
        y_train = df.loc[train_mask, target_col].values.astype(float)
        X_test = df.loc[test_mask, feature_cols].values.astype(float)

        # --- Uniform baseline ---
        # Predict the mean of training zone targets
        train_zones = df.loc[train_mask, zone_col].unique()
        train_targets = [df.loc[valid_mask & (df[zone_col] == z), target_col].iloc[0] for z in train_zones]
        uniform_pred = np.mean(train_targets)
        rows.append({
            "zone": held_out_zone, "method": "uniform",
            "target": target_value, "predicted_aggregate": uniform_pred,
            "abs_error": abs(uniform_pred - target_value),
        })

        # --- RWI baseline ---
        rwi_test = df.loc[test_mask, "rwi"].values.astype(float)
        raw_scores = np.exp(-rwi_test)
        if test_pop.sum() > 0:
            rwi_weighted_mean = np.average(raw_scores, weights=test_pop)
        else:
            rwi_weighted_mean = raw_scores.mean()
        # Use mean of training targets as best guess for held-out zone
        rwi_pred_agg = np.mean(train_targets)
        rows.append({
            "zone": held_out_zone, "method": "rwi",
            "target": target_value, "predicted_aggregate": rwi_pred_agg,
            "abs_error": abs(rwi_pred_agg - target_value),
        })

        # --- Ridge ---
        ridge_cfg = cfg["modeling"]["ridge"]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(
                alphas=ridge_cfg["alpha_candidates"],
                cv=min(ridge_cfg["cv_folds"], len(np.unique(y_train))),
                scoring="neg_mean_squared_error",
            )),
        ])
        pipe.fit(X_train, y_train)
        ridge_preds = pipe.predict(X_test)
        if test_pop.sum() > 0:
            ridge_agg = np.average(ridge_preds, weights=test_pop)
        else:
            ridge_agg = ridge_preds.mean()
        rows.append({
            "zone": held_out_zone, "method": "ridge",
            "target": target_value, "predicted_aggregate": ridge_agg,
            "abs_error": abs(ridge_agg - target_value),
        })

        # --- GBM (optional) ---
        if skip_gbm:
            logger.debug("GBM LOZO skipped (--skip-gbm).")
        try:
            if skip_gbm:
                raise ImportError("skip_gbm set")
            from src.models.gbm_model import _get_gbm_model
            gbm = _get_gbm_model(cfg)
            gbm.fit(X_train, y_train)
            gbm_preds = gbm.predict(X_test)
            if test_pop.sum() > 0:
                gbm_agg = np.average(gbm_preds, weights=test_pop)
            else:
                gbm_agg = gbm_preds.mean()
            rows.append({
                "zone": held_out_zone, "method": "gbm",
                "target": target_value, "predicted_aggregate": gbm_agg,
                "abs_error": abs(gbm_agg - target_value),
            })
        except Exception as e:
            logger.debug("GBM LOZO skipped: %s", e)

        # --- GAM (optional) ---
        try:
            if skip_gam:
                raise ImportError("skip_gam set")
            from src.models.gam_model import GAMDeprivationModel
            gam_cfg = cfg["modeling"].get("gam", {})
            gam = GAMDeprivationModel(
                n_splines=gam_cfg.get("n_splines", 10),
                lam_candidates=gam_cfg.get("lam_candidates", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
                max_iter=gam_cfg.get("max_iter", 100),
            )
            gam.fit(X_train, y_train, feature_names=feature_cols)
            gam_preds = gam.predict(X_test)
            if test_pop.sum() > 0:
                gam_agg = np.average(gam_preds, weights=test_pop)
            else:
                gam_agg = gam_preds.mean()
            rows.append({
                "zone": held_out_zone, "method": "gam",
                "target": target_value, "predicted_aggregate": gam_agg,
                "abs_error": abs(gam_agg - target_value),
            })
        except Exception as e:
            logger.debug("GAM LOZO skipped: %s", e)

        # --- WSNN (optional — requires PyTorch) ---
        try:
            if skip_wsnn:
                raise ImportError("skip_wsnn set")
            from src.models.weakly_supervised_nn import (
                WeaklySupervisedNN, _prepare_zone_groups,
            )
            from sklearn.impute import SimpleImputer

            wsnn_cfg = cfg["modeling"].get("weakly_supervised", {})

            # Impute NaN features for train/test
            imp = SimpleImputer(strategy="median")
            X_train_imp = imp.fit_transform(X_train)
            X_test_imp = imp.transform(X_test)

            # Build zone-level supervision groups from training zones only
            train_df = df.loc[train_mask].copy()
            train_groups, train_targets = _prepare_zone_groups(train_df, target_col)
            train_pop = train_df[pop_col].values.astype(float)
            train_pop = np.where(np.isnan(train_pop) | (train_pop < 0), 0.0, train_pop)

            wsnn = WeaklySupervisedNN(
                input_dim=X_train_imp.shape[1],
                hidden_dims=wsnn_cfg.get("hidden_dims", [64, 32]),
                learning_rate=wsnn_cfg.get("learning_rate", 0.001),
                dropout=wsnn_cfg.get("dropout", 0.2),
            )
            wsnn.fit(
                X_train_imp, train_pop, train_groups, train_targets,
                n_epochs=wsnn_cfg.get("n_epochs", 100),
                verbose=False,
            )

            # Predict on held-out zone WITHOUT reconciliation
            wsnn_preds = wsnn.predict(X_test_imp)
            if test_pop.sum() > 0:
                wsnn_agg = np.average(wsnn_preds, weights=test_pop)
            else:
                wsnn_agg = wsnn_preds.mean()
            rows.append({
                "zone": held_out_zone, "method": "wsnn",
                "target": target_value, "predicted_aggregate": wsnn_agg,
                "abs_error": abs(wsnn_agg - target_value),
            })
        except Exception as e:
            logger.debug("WSNN LOZO skipped: %s", e)

    result = pd.DataFrame(rows)
    if len(result) > 0:
        logger.info("LOZO results:\n%s", result.to_string(index=False))
    return result
