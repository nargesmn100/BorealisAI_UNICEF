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
            y_train = df.loc[train_mask, "quintile_target_moderate"].values.astype(float)
            logger.info(
                "Using quintile pseudo-targets for training (%d cells). "
                "Range: [%.1f%%, %.1f%%]",
                len(y_train), np.nanmin(y_train), np.nanmax(y_train),
            )
        else:
            y_train = df.loc[train_mask, target_moderate].values.astype(float)
            logger.info("Quintile targets insufficient. Falling back to zone-level targets.")
    else:
        y_train = df.loc[train_mask, target_moderate].values.astype(float)
        if use_quintile_target:
            logger.info("quintile_target_moderate column not found. Using zone-level targets.")

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
        y_train = y_train - rwi_prior_train
        logger.info(
            "RWI prior computed. Residual range: [%.3f, %.3f], mean=%.4f",
            y_train.min(), y_train.max(), y_train.mean(),
        )
    else:
        if use_rwi_prior:
            logger.info("RWI column not found. Skipping RWI prior.")

    # ------------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------------
    logger.info("Fitting Ridge regression model...")
    model = RidgeDeprivationModel(
        alpha_candidates=ridge_cfg["alpha_candidates"],
        cv_folds=ridge_cfg["cv_folds"],
        random_state=ridge_cfg["random_state"],
    )
    model.fit(X_train, y_train, feature_names=feature_cols)

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
    _, lower_raw, upper_raw = bootstrap_uncertainty(
        model_cls=RidgeDeprivationModel,
        model_kwargs={
            "alpha_candidates": ridge_cfg["alpha_candidates"],
            "cv_folds": ridge_cfg["cv_folds"],
            "random_state": rs,
        },
        X_train=X_train,
        y_train=y_train,
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
