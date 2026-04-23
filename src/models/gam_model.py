"""
Generalised Additive Model (GAM) — Interpretable Spatial Deprivation Model

Scientific framing
------------------
A GAM fits a sum of smooth univariate spline functions, one per feature:

    y_hat = beta_0 + f_1(rwi) + f_2(population) + f_3(travel_time) + ...

This is strictly more interpretable than GBM (individual smooth curves can be
plotted per feature) while capturing mild nonlinearities that Ridge misses.

Implementation
--------------
Uses `pygam` (pip install pygam).  If pygam is not installed, the module raises
a clear ImportError at import time with install instructions.

We use LinearGAM with automatic lambda selection via grid search GCV.
Splines are fitted for continuous features; binary/categorical features use
linear terms.

Model outputs
-------------
  gam_raw_score        : raw GAM prediction (before reconciliation)
  gam_moderate         : reconciled moderate prevalence
  gam_severe           : reconciled severe prevalence
  gam_moderate_lower   : lower 90% CI from GAM's built-in confidence bands
  gam_moderate_upper   : upper 90% CI from GAM's built-in confidence bands
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    from pygam import LinearGAM, s, l, f
    _PYGAM_AVAILABLE = True
except ImportError:
    _PYGAM_AVAILABLE = False

from src.reconciliation.admin_reconcile import reconcile_predictions, reconcile_uncertainty_bounds
from src.utils.config_loader import get_available_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature term builder
# ---------------------------------------------------------------------------

# Features that should use spline terms (continuous)
_SPLINE_FEATURES = {
    "rwi", "population", "log_population",
    "travel_time_cities", "log_travel_time_cities",
    "travel_time_50k", "log_travel_time_50k",
}

# Features that should use linear terms (binary / integer class codes)
_LINEAR_FEATURES = {"is_urban", "smod_class"}


def _build_gam_terms(feature_names: list):
    """
    Build a pygam term list from feature names.

    Continuous features → spline terms s(i, n_splines=10)
    Binary / class features → linear terms l(i)

    Parameters
    ----------
    feature_names : list of str

    Returns
    -------
    pygam term object (sum of terms)
    """
    terms = None
    for i, name in enumerate(feature_names):
        if name in _LINEAR_FEATURES:
            term = l(i)
        else:
            term = s(i, n_splines=10, constraints="monotonic_dec"
                     if name == "rwi" else None)
        terms = term if terms is None else terms + term
    return terms


# ---------------------------------------------------------------------------
# GAM model class
# ---------------------------------------------------------------------------

class GAMDeprivationModel:
    """
    Generalised Additive Model for spatial deprivation estimation.

    Wraps pygam.LinearGAM with automatic lambda (smoothness) selection
    via generalised cross-validation (GCV).

    Parameters
    ----------
    n_splines : int
        Number of spline basis functions per continuous feature.
    lam_candidates : list of float
        Smoothing parameter candidates for GCV search.
    max_iter : int
        Maximum PIRLS iterations.
    random_state : int
        Used only for reproducibility of any stochastic steps.
    """

    def __init__(
        self,
        n_splines: int = 10,
        lam_candidates: list = None,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        if not _PYGAM_AVAILABLE:
            raise ImportError(
                "pygam is required for the GAM model. "
                "Install it with: pip install pygam"
            )
        self.n_splines = n_splines
        self.lam_candidates = lam_candidates or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.max_iter = max_iter
        self.random_state = random_state
        self.model: Optional[LinearGAM] = None
        self.feature_names: list = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
    ) -> "GAMDeprivationModel":
        """
        Fit the GAM model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        feature_names : list or None

        Returns
        -------
        self
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        terms = _build_gam_terms(self.feature_names)
        self.model = LinearGAM(terms, max_iter=self.max_iter)

        logger.info("Fitting GAM with GCV lambda search over %d candidates...",
                    len(self.lam_candidates))
        self.model.gridsearch(X_arr, y, lam=self.lam_candidates, progress=False)

        logger.info("GAM fitted. Best lam: %s", self.model.lam)
        logger.info("GAM pseudo-R²: %.4f", self.model.statistics_["pseudo_r2"]["McFadden"])
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Predict raw deprivation scores.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict(X_arr)

    def predict_intervals(
        self,
        X: pd.DataFrame | np.ndarray,
        width: float = 0.90,
    ) -> tuple:
        """
        Return prediction confidence intervals from GAM's built-in bands.

        Parameters
        ----------
        X : array-like
        width : float
            Confidence level (default 0.90 → 5th–95th percentiles).

        Returns
        -------
        (lower, upper) : (np.ndarray, np.ndarray)
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        lower, upper = self.model.prediction_intervals(X_arr, width=width).T
        return lower, upper

    def get_partial_dependence(self) -> dict:
        """
        Extract partial dependence data for each feature (for plotting).

        Returns
        -------
        dict : {feature_name: {"x": array, "y": array, "lower": array, "upper": array}}
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        result = {}
        for i, name in enumerate(self.feature_names):
            try:
                xx = self.model.generate_X_grid(term=i)
                pdep, confi = self.model.partial_dependence(term=i, X=xx, width=0.95)
                result[name] = {
                    "x": xx[:, i],
                    "y": pdep,
                    "lower": confi[:, 0],
                    "upper": confi[:, 1],
                }
            except Exception as e:
                logger.debug("Could not compute partial dependence for '%s': %s", name, e)
        return result


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run(
    cfg: dict,
    df: pd.DataFrame,
) -> tuple:
    """
    Fit the GAM model, generate predictions, reconcile, and add uncertainty.

    Parameters
    ----------
    cfg : dict
        Loaded config.
    df : pd.DataFrame
        Modeling table (output of merge_features step).

    Returns
    -------
    (pd.DataFrame, GAMDeprivationModel)
        Enriched DataFrame and the fitted GAM model.
    """
    if not _PYGAM_AVAILABLE:
        raise ImportError(
            "pygam is required for the GAM model. "
            "Install it with: pip install pygam"
        )

    feature_cols = get_available_features(cfg, df)
    target_moderate = "moderate_prevalence"
    target_severe = "severe_prevalence"
    zone_col = cfg["modeling"]["admin_zone_col"]
    pop_col = "population"
    gam_cfg = cfg["modeling"].get("gam", {})

    # ------------------------------------------------------------------
    # Prepare training data
    # ------------------------------------------------------------------
    model_mask = df["in_modeling_sample"].fillna(False)
    feature_mask = df[feature_cols].notna().all(axis=1)
    train_mask = model_mask & feature_mask

    logger.info(
        "GAM training: %d cells with complete features out of %d in modeling sample.",
        train_mask.sum(), model_mask.sum(),
    )

    if train_mask.sum() < 10:
        raise ValueError(
            f"Only {train_mask.sum()} training samples — too few to fit GAM. "
            "Check data pipeline."
        )

    X_train = df.loc[train_mask, feature_cols].values.astype(float)
    y_train = df.loc[train_mask, target_moderate].values.astype(float)

    # ------------------------------------------------------------------
    # Fit GAM
    # ------------------------------------------------------------------
    logger.info("Fitting GAM model...")
    n_splines = gam_cfg.get("n_splines", 10)
    lam_candidates = gam_cfg.get("lam_candidates", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    max_iter = gam_cfg.get("max_iter", 100)
    random_state = gam_cfg.get("random_state", 42)

    model = GAMDeprivationModel(
        n_splines=n_splines,
        lam_candidates=lam_candidates,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train, feature_names=feature_cols)

    # ------------------------------------------------------------------
    # Predict on all valid cells
    # ------------------------------------------------------------------
    pred_mask = feature_mask & (df["subregion"] != "Unknown")
    X_pred = df.loc[pred_mask, feature_cols].values.astype(float)

    raw_preds = model.predict(X_pred)

    clip_min = gam_cfg.get("clip_raw_min")
    clip_max = gam_cfg.get("clip_raw_max")
    if clip_min is not None or clip_max is not None:
        lo = clip_min if clip_min is not None else -np.inf
        hi = clip_max if clip_max is not None else np.inf
        raw_preds = np.clip(raw_preds.astype(float), lo, hi)
        logger.info("GAM raw scores clipped to [%.4f, %.4f] before reconciliation.", lo, hi)

    df = df.copy()
    df["gam_raw_score"] = np.nan
    df.loc[pred_mask, "gam_raw_score"] = raw_preds

    # ------------------------------------------------------------------
    # Reconcile moderate predictions
    # ------------------------------------------------------------------
    logger.info("Reconciling GAM predictions to moderate prevalence targets...")
    df = reconcile_predictions(
        df,
        raw_score_col="gam_raw_score",
        target_col=target_moderate,
        zone_col=zone_col,
        population_col=pop_col,
        output_col="gam_moderate",
        strategy="population_weighted",
    )

    # ------------------------------------------------------------------
    # Reconcile severe predictions
    # ------------------------------------------------------------------
    logger.info("Reconciling GAM predictions to severe prevalence targets...")
    df = reconcile_predictions(
        df,
        raw_score_col="gam_raw_score",
        target_col=target_severe,
        zone_col=zone_col,
        population_col=pop_col,
        output_col="gam_severe",
        strategy="population_weighted",
    )

    # ------------------------------------------------------------------
    # Reconcile depth metrics (same raw score, different targets)
    # ------------------------------------------------------------------
    if "moderate_depth" in df.columns:
        logger.info("Reconciling GAM predictions to depth targets...")
        df = reconcile_predictions(
            df, raw_score_col="gam_raw_score", target_col="moderate_depth",
            zone_col=zone_col, population_col=pop_col,
            output_col="gam_moderate_depth", strategy="population_weighted",
        )
        df = reconcile_predictions(
            df, raw_score_col="gam_raw_score", target_col="severe_depth",
            zone_col=zone_col, population_col=pop_col,
            output_col="gam_severe_depth", strategy="population_weighted",
        )

    # ------------------------------------------------------------------
    # Confidence intervals from GAM's built-in bands
    # ------------------------------------------------------------------
    logger.info("Computing GAM prediction intervals (90%% CI)...")
    try:
        lower_raw, upper_raw = model.predict_intervals(X_pred, width=0.90)
        df["gam_moderate_lower"] = np.nan
        df["gam_moderate_upper"] = np.nan
        df.loc[pred_mask, "gam_moderate_lower"] = lower_raw
        df.loc[pred_mask, "gam_moderate_upper"] = upper_raw

        # Propagate uncertainty bounds through reconciliation
        df = reconcile_uncertainty_bounds(
            df,
            lower_col="gam_moderate_lower",
            upper_col="gam_moderate_upper",
            raw_col="gam_raw_score",
            reconciled_col="gam_moderate",
            zone_col=zone_col,
            population_col=pop_col,
        )
    except Exception as e:
        logger.warning("Could not compute GAM prediction intervals: %s", e)

    logger.info("GAM model predictions complete.")
    return df, model
