"""
Ridge Regression Model — Interpretable Baseline ML Model

Scientific framing
------------------
We do NOT train the model to directly predict the official zone-level
prevalence values (that would be overfitting to only 3 target values).

Instead, the model learns a *relative spatial allocation function*:
  y_hat_i = f(x_i)

where y_hat_i is an unconstrained deprivation score for cell i.
After prediction, post-processing (reconciliation) rescales the outputs
so they match the official zone totals.

Training setup
--------------
Since we only have 3 zone-level targets (Urban/Rural/KMA), we use a
*pseudo-regression* approach:
- Each grid cell is assigned the zone-level target as its "label"
- The model learns to predict a smooth function that varies within zones
  based on the proxy features
- Cross-validation is done across cells (not zones) to select alpha

This is a common approach when ground truth is available only at aggregate
level (weak supervision / area-level regression).

Limitations
-----------
- Training labels are the same within each zone — the model must learn
  spatial variation from feature correlations alone
- With only 3 zones, the model cannot learn zone-specific intercepts
- Ridge regression gives linear effects of features — interpretable via
  coefficients and standardised importances
- Optional *stacked* DHS cluster term (``dhs_aux_dhs_scale``): extra weighted
  least-squares rows that align ``Xβ`` with nearest-cluster DHS deprivation
  in addition to the MICS/zone (or soft-blend) target — distinct from the scalar
  DHS soft-label blend.

Model outputs
-------------
  ridge_raw_score         : raw model prediction (before reconciliation)
  ridge_moderate          : reconciled moderate prevalence prediction
  ridge_severe            : reconciled severe prevalence prediction
  ridge_moderate_upper    : upper uncertainty bound (bootstrap)
  ridge_moderate_lower    : lower uncertainty bound (bootstrap)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from src.reconciliation.admin_reconcile import reconcile_predictions, reconcile_uncertainty_bounds
from src.utils.config_loader import get_available_features

logger = logging.getLogger(__name__)


class RidgeDeprivationModel:
    """
    Interpretable ridge regression model for spatial deprivation estimation.

    Parameters
    ----------
    alpha_candidates : list of float
        Regularisation strengths to try in cross-validation.
    cv_folds : int
        Number of CV folds.
    random_state : int
    """

    def __init__(
        self,
        alpha_candidates: list = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.alpha_candidates = alpha_candidates or [0.01, 0.1, 1.0, 10.0, 100.0]
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: list = []
        self.best_alpha: Optional[float] = None

    def _build_pipeline(self) -> Pipeline:
        """Construct sklearn Pipeline with StandardScaler + RidgeCV."""
        ridge = RidgeCV(
            alphas=self.alpha_candidates,
            cv=self.cv_folds,
            scoring="neg_mean_squared_error",
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", ridge),
        ])

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "RidgeDeprivationModel":
        """
        Fit the ridge regression model.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array (zone-level poverty prevalence for each cell).
        feature_names : list or None
            Feature column names for logging.

        Returns
        -------
        self
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_arr, y)

        self.best_alpha = self.pipeline.named_steps["ridge"].alpha_
        logger.info("Ridge model fitted. Best alpha: %.4f", self.best_alpha)

        # Log feature coefficients (standardised)
        scaler = self.pipeline.named_steps["scaler"]
        coefs = self.pipeline.named_steps["ridge"].coef_
        std = scaler.scale_

        logger.info("Feature importances (standardised coefficients):")
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        for i in sorted_idx:
            name = self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
            logger.info("  %-35s : %+.4f  (raw coef: %+.4f, std: %.4f)",
                        name, coefs[i] * std[i], coefs[i], std[i])

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict raw deprivation scores.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray

        Returns
        -------
        np.ndarray
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict(X_arr)

    def get_coef_table(self) -> pd.DataFrame:
        """
        Return a DataFrame of feature names, raw coefficients, and standardised impacts.

        Returns
        -------
        pd.DataFrame
        """
        if self.pipeline is None:
            raise RuntimeError("Model not fitted.")
        scaler = self.pipeline.named_steps["scaler"]
        coefs = self.pipeline.named_steps["ridge"].coef_
        std = scaler.scale_
        return pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": coefs,
            "standardised_impact": coefs * std,
        }).sort_values("standardised_impact", key=abs, ascending=False).reset_index(drop=True)


def bootstrap_uncertainty(
    model_cls,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    n_bootstrap: int = 50,
    random_state: int = 42,
) -> tuple:
    """
    Estimate prediction uncertainty via bootstrap resampling.

    Parameters
    ----------
    model_cls : class
        Model class (e.g. RidgeDeprivationModel).
    model_kwargs : dict
        Keyword arguments to pass to model_cls.
    X_train, y_train : array-like
        Training data.
    X_pred : array-like
        Prediction features.
    n_bootstrap : int
    random_state : int

    Returns
    -------
    (mean, lower, upper) : (np.ndarray, np.ndarray, np.ndarray)
        Mean prediction and 5th/95th percentile uncertainty bounds.
    """
    rng = np.random.default_rng(random_state)
    preds = np.zeros((n_bootstrap, len(X_pred)))

    logger.info("Running %d bootstrap samples for uncertainty estimation...", n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        X_b = X_train[idx]
        y_b = y_train[idx]
        m = model_cls(**model_kwargs)
        m.fit(X_b, y_b)
        preds[b] = m.predict(X_pred)

    mean_pred = preds.mean(axis=0)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)

    logger.info(
        "Bootstrap uncertainty: mean range [%.3f, %.3f], "
        "90%% CI width: mean=%.3f",
        mean_pred.min(), mean_pred.max(),
        (upper - lower).mean(),
    )

    return mean_pred, lower, upper


def build_dhs_stacked_ridge_xy(
    X: np.ndarray,
    y_mics_residual: np.ndarray,
    y_dhs_residual: np.ndarray,
    dhs_valid: np.ndarray,
    mics_scale: float,
    dhs_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stacked (weighted) least squares: augment the MICS block with rows that
    pull predictions toward DHS cluster deprivation on cells with a valid
    nearest-cluster index.

    Each block is scaled by sqrt(weight) so the effective loss is
    w_mics * ||y_mics - Xβ||² + w_dhs * ||y_dhs - Xβ||² on the DHS-labeled rows.
    """
    sm = float(np.sqrt(max(mics_scale, 0.0)))
    sd = float(np.sqrt(max(dhs_scale, 0.0)))
    n = len(X)
    y_dhs = np.asarray(y_dhs_residual, dtype=float)
    valid = np.asarray(dhs_valid, dtype=bool) & np.isfinite(y_dhs)
    if sm <= 0.0 or n == 0:
        raise ValueError("Invalid MICS scale or empty training set for stacked Ridge.")
    X_top = sm * X
    y_top = sm * np.asarray(y_mics_residual, dtype=float)
    if sd <= 0.0 or not valid.any():
        return X_top, y_top
    X_bot = sd * X[valid]
    y_bot = sd * y_dhs[valid]
    return np.vstack([X_top, X_bot]), np.concatenate([y_top, y_bot])


def bootstrap_uncertainty_dhs_stack(
    model_cls,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_mics_res: np.ndarray,
    y_dhs_res: np.ndarray,
    dhs_valid: np.ndarray,
    mics_scale: float,
    dhs_scale: float,
    X_pred: np.ndarray,
    n_bootstrap: int = 50,
    random_state: int = 42,
) -> tuple:
    """
    Bootstrap for Ridge when training used DHS stacked targets; resamples
    the MICS row index and rebuilds the stacked matrix each draw.
    """
    rng = np.random.default_rng(random_state)
    n = len(X_train)
    preds = np.zeros((n_bootstrap, len(X_pred)))
    dhs_v = np.asarray(dhs_valid, dtype=bool)
    y_m = np.asarray(y_mics_res, dtype=float)
    y_d = np.asarray(y_dhs_res, dtype=float)

    logger.info("Running %d bootstrap samples (DHS-stacked Ridge)...", n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_s, y_s = build_dhs_stacked_ridge_xy(
            X_train[idx],
            y_m[idx],
            y_d[idx],
            dhs_v[idx],
            mics_scale,
            dhs_scale,
        )
        m = model_cls(**model_kwargs)
        m.fit(X_s, y_s)
        preds[b] = m.predict(X_pred)

    mean_pred = preds.mean(axis=0)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)
    logger.info(
        "Bootstrap uncertainty (DHS stack): mean range [%.3f, %.3f], "
        "90%% CI width: mean=%.3f",
        mean_pred.min(), mean_pred.max(),
        (upper - lower).mean(),
    )
    return mean_pred, lower, upper


def run(
    cfg: dict,
    df: pd.DataFrame,
) -> tuple:
    """
    Fit the ridge model, generate predictions, reconcile, and add uncertainty.

    Parameters
    ----------
    cfg : dict
    df : pd.DataFrame
        Modeling table.

    Returns
    -------
    (pd.DataFrame, RidgeDeprivationModel)
        Enriched DataFrame and the fitted model.
    """
    feature_cols = get_available_features(cfg, df)
    # Use the renamed column names from the modeling table (not raw Excel names)
    target_moderate = "moderate_prevalence"
    target_severe = "severe_prevalence"
    zone_col = cfg["modeling"]["admin_zone_col"]
    pop_col = "population"
    ridge_cfg = cfg["modeling"]["ridge"]
    uncertainty_cfg = cfg["modeling"]["uncertainty"]

    use_rwi_prior = ridge_cfg.get("use_rwi_prior", False)
    use_quintile_target = ridge_cfg.get("use_quintile_target", False)

    # ------------------------------------------------------------------
    # Prepare training data
    # Filter to cells in the modeling sample
    # ------------------------------------------------------------------
    model_mask = df["in_modeling_sample"].fillna(False)

    # Drop rows with any missing feature values
    feature_mask = df[feature_cols].notna().all(axis=1)
    train_mask = model_mask & feature_mask

    logger.info(
        "Ridge training: %d cells with complete features out of %d in modeling sample.",
        train_mask.sum(), model_mask.sum(),
    )

    if train_mask.sum() < 10:
        raise ValueError(
            f"Only {train_mask.sum()} training samples available after filtering. "
            "Cannot fit a reliable model. Check data pipeline."
        )

    X_train = df.loc[train_mask, feature_cols].values.astype(float)

    # ------------------------------------------------------------------
    # Select training target
    # ------------------------------------------------------------------
    if use_quintile_target and "quintile_target_moderate" in df.columns:
        qt_valid = df.loc[train_mask, "quintile_target_moderate"].notna()
        if qt_valid.sum() > 10:
            y_mics = df.loc[train_mask, "quintile_target_moderate"].values.astype(float)
            logger.info(
                "Using quintile pseudo-targets for training (%d cells). "
                "Range: [%.1f%%, %.1f%%]",
                len(y_mics), np.nanmin(y_mics), np.nanmax(y_mics),
            )
        else:
            y_mics = df.loc[train_mask, target_moderate].values.astype(float)
            logger.info("Quintile targets insufficient. Falling back to zone-level targets.")
    else:
        y_mics = df.loc[train_mask, target_moderate].values.astype(float)
        if use_quintile_target:
            logger.info("quintile_target_moderate column not found. Using zone-level targets.")

    # DHS: cluster-level target (0–1 → %) aligned with MICS / Ridge label scale
    dhs_train_pct = None
    if "dhs_nearest_dep_index" in df.columns:
        dhs_train_pct = (
            df.loc[train_mask, "dhs_nearest_dep_index"].to_numpy(dtype=float) * 100.0
        )
    dhs_aux_dhs = float(ridge_cfg.get("dhs_aux_dhs_scale", 0.0))
    dhs_aux_mics = float(ridge_cfg.get("dhs_aux_mics_scale", 1.0))
    use_dhs_stack = (
        dhs_aux_dhs > 0.0
        and dhs_train_pct is not None
    )

    # ------------------------------------------------------------------
    # Optional DHS soft label: blend aggregate target with nearest-cluster DHS
    # (skipped when DHS stacked auxiliary term is used — that term carries cluster signal)
    # ------------------------------------------------------------------
    use_dhs_sl = ridge_cfg.get("use_dhs_soft_label", False)
    dhs_w = float(ridge_cfg.get("dhs_soft_label_weight", 0.0))
    if use_dhs_stack:
        y_train = y_mics.copy()
        if use_dhs_sl and dhs_w > 0:
            logger.info(
                "DHS soft-label blend is disabled when dhs_aux_dhs_scale > 0 "
                "(using stacked cluster supervision instead).",
            )
    elif use_dhs_sl and dhs_w > 0 and dhs_train_pct is not None:
        dhs_pct = dhs_train_pct
        valid = np.isfinite(dhs_pct)
        if valid.sum() > 0:
            y_zone = y_mics.copy()
            y_train = y_zone.copy()
            y_train[valid] = (1.0 - dhs_w) * y_zone[valid] + dhs_w * dhs_pct[valid]
            logger.info(
                "DHS soft label blend (weight=%.3f): using %d cells with nearest-cluster DHS.",
                dhs_w,
                int(valid.sum()),
            )
        else:
            y_train = y_mics.copy()
    else:
        y_train = y_mics.copy()

    # ------------------------------------------------------------------
    # RWI prior: compute prior and train on residuals
    # ------------------------------------------------------------------
    rwi_prior_train = None
    rwi_prior_pred = None

    if use_rwi_prior and "rwi" in df.columns:
        logger.info("Computing RWI-based prior for Ridge residual training...")
        rwi_all = df["rwi"].values.astype(float)

        # Prior: exp(-rwi) — higher deprivation for lower wealth
        raw_prior = np.exp(-rwi_all)

        # Normalize per zone so zone-level weighted mean matches zone target
        pred_mask_temp = feature_mask & (df["subregion"] != "Unknown")
        rwi_prior_full = np.full(len(df), np.nan)

        for zone in df.loc[pred_mask_temp, zone_col].unique():
            zmask = pred_mask_temp & (df[zone_col] == zone)
            zone_prior = raw_prior[zmask]
            zone_pop = df.loc[zmask, pop_col].values.astype(float)
            zone_pop = np.where(np.isnan(zone_pop) | (zone_pop <= 0), 1e-6, zone_pop)
            zone_target = df.loc[zmask, target_moderate].iloc[0]

            # Rescale so pop-weighted mean = zone target
            weighted_mean = np.average(zone_prior, weights=zone_pop)
            if weighted_mean > 0:
                rwi_prior_full[zmask] = zone_prior * (zone_target / weighted_mean)
            else:
                rwi_prior_full[zmask] = zone_target

        rwi_prior_train = rwi_prior_full[train_mask]
        y_mics_res = y_train - rwi_prior_train
        logger.info(
            "RWI prior computed. MICS residual range: [%.3f, %.3f], mean=%.4f",
            y_mics_res.min(), y_mics_res.max(), y_mics_res.mean(),
        )
    else:
        y_mics_res = y_train
        if use_rwi_prior:
            logger.info("RWI column not found. Skipping RWI prior.")

    y_dhs_res = None
    dhs_valid = None
    if dhs_train_pct is not None:
        y_dhs_res = dhs_train_pct.astype(float).copy()
        if rwi_prior_train is not None:
            y_dhs_res = y_dhs_res - rwi_prior_train
        dhs_valid = np.isfinite(dhs_train_pct) & np.isfinite(y_dhs_res)
    if use_dhs_stack:
        if dhs_valid is None or not dhs_valid.any():
            logger.warning(
                "dhs_aux_dhs_scale > 0 but no valid DHS cluster labels on training rows. "
                "Fitting MICS targets only (no DHS stack).",
            )
            use_dhs_stack = False
        else:
            logger.info(
                "DHS stacked Ridge: mics_scale=%.4f, dhs_scale=%.4f, "
                "DHS-supervised training rows in aux block: %d.",
                dhs_aux_mics,
                dhs_aux_dhs,
                int(dhs_valid.sum()),
            )

    X_train_fit = X_train
    y_train_fit = y_mics_res
    if use_dhs_stack:
        X_train_fit, y_train_fit = build_dhs_stacked_ridge_xy(
            X_train,
            y_mics_res,
            y_dhs_res,
            dhs_valid,
            dhs_aux_mics,
            dhs_aux_dhs,
        )

    # ------------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------------
    logger.info("Fitting Ridge regression model...")
    model = RidgeDeprivationModel(
        alpha_candidates=ridge_cfg["alpha_candidates"],
        cv_folds=ridge_cfg["cv_folds"],
        random_state=ridge_cfg["random_state"],
    )
    model.fit(X_train_fit, y_train_fit, feature_names=feature_cols)

    # Log coefficient table
    coef_table = model.get_coef_table()
    logger.info("Ridge coefficient table:\n%s", coef_table.to_string(index=False))

    # ------------------------------------------------------------------
    # Predict on all valid cells
    # ------------------------------------------------------------------
    pred_mask = feature_mask & (df["subregion"] != "Unknown")
    X_pred = df.loc[pred_mask, feature_cols].values.astype(float)

    raw_preds = model.predict(X_pred)

    # Add RWI prior back to get final raw scores
    if rwi_prior_train is not None:
        rwi_prior_pred = rwi_prior_full[pred_mask]
        raw_preds = raw_preds + rwi_prior_pred
        logger.info(
            "RWI prior added back. Final raw prediction range: [%.3f, %.3f]",
            raw_preds.min(), raw_preds.max(),
        )

    df = df.copy()
    df["ridge_raw_score"] = np.nan
    df.loc[pred_mask, "ridge_raw_score"] = raw_preds

    # ------------------------------------------------------------------
    # Reconcile moderate predictions
    # ------------------------------------------------------------------
    logger.info("Reconciling ridge predictions to moderate prevalence targets...")
    df = reconcile_predictions(
        df,
        raw_score_col="ridge_raw_score",
        target_col=target_moderate,
        zone_col=zone_col,
        population_col=pop_col,
        output_col="ridge_moderate",
        strategy="population_weighted",
    )

    # ------------------------------------------------------------------
    # Reconcile severe predictions
    # ------------------------------------------------------------------
    logger.info("Reconciling ridge predictions to severe prevalence targets...")
    df = reconcile_predictions(
        df,
        raw_score_col="ridge_raw_score",
        target_col=target_severe,
        zone_col=zone_col,
        population_col=pop_col,
        output_col="ridge_severe",
        strategy="population_weighted",
    )

    # ------------------------------------------------------------------
    # Reconcile depth metrics (same raw score, different targets)
    # ------------------------------------------------------------------
    if "moderate_depth" in df.columns:
        logger.info("Reconciling ridge predictions to depth targets...")
        df = reconcile_predictions(
            df, raw_score_col="ridge_raw_score", target_col="moderate_depth",
            zone_col=zone_col, population_col=pop_col,
            output_col="ridge_moderate_depth", strategy="population_weighted",
        )
        df = reconcile_predictions(
            df, raw_score_col="ridge_raw_score", target_col="severe_depth",
            zone_col=zone_col, population_col=pop_col,
            output_col="ridge_severe_depth", strategy="population_weighted",
        )

    # ------------------------------------------------------------------
    # Bootstrap uncertainty estimation
    # ------------------------------------------------------------------
    n_bootstrap = uncertainty_cfg.get("n_bootstrap", 50)
    rs = uncertainty_cfg.get("random_state", 42)

    logger.info("Running bootstrap uncertainty estimation (%d samples)...", n_bootstrap)
    if use_dhs_stack and dhs_valid is not None and y_dhs_res is not None:
        _, lower_raw, upper_raw = bootstrap_uncertainty_dhs_stack(
            model_cls=RidgeDeprivationModel,
            model_kwargs={
                "alpha_candidates": ridge_cfg["alpha_candidates"],
                "cv_folds": ridge_cfg["cv_folds"],
                "random_state": rs,
            },
            X_train=X_train,
            y_mics_res=y_mics_res,
            y_dhs_res=y_dhs_res,
            dhs_valid=dhs_valid,
            mics_scale=dhs_aux_mics,
            dhs_scale=dhs_aux_dhs,
            X_pred=X_pred,
            n_bootstrap=n_bootstrap,
            random_state=rs,
        )
    else:
        _, lower_raw, upper_raw = bootstrap_uncertainty(
            model_cls=RidgeDeprivationModel,
            model_kwargs={
                "alpha_candidates": ridge_cfg["alpha_candidates"],
                "cv_folds": ridge_cfg["cv_folds"],
                "random_state": rs,
            },
            X_train=X_train,
            y_train=y_mics_res,
            X_pred=X_pred,
            n_bootstrap=n_bootstrap,
            random_state=rs,
        )

    df["ridge_moderate_lower"] = np.nan
    df["ridge_moderate_upper"] = np.nan
    df.loc[pred_mask, "ridge_moderate_lower"] = lower_raw
    df.loc[pred_mask, "ridge_moderate_upper"] = upper_raw

    # Propagate uncertainty bounds through reconciliation
    df = reconcile_uncertainty_bounds(
        df,
        lower_col="ridge_moderate_lower",
        upper_col="ridge_moderate_upper",
        raw_col="ridge_raw_score",
        reconciled_col="ridge_moderate",
        zone_col=zone_col,
        population_col=pop_col,
    )

    logger.info("Ridge model predictions complete.")
    return df, model
