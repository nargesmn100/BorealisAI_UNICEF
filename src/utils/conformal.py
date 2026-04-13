"""
Split-Conformal Prediction Intervals

Provides marginal coverage guarantees without distributional assumptions:

    P(y ∈ [ŷ − q̂, ŷ + q̂]) ≥ 1 − α

Method: Venn-Shafer / split-conformal (Papadopoulos et al. 2002;
Angelopoulos & Bates 2022 "A Gentle Introduction to Conformal Prediction").

Usage
-----
1. Hold out a calibration split from training data.
2. Fit the model on the remaining training data.
3. Call `.calibrate(y_cal, y_hat_cal)` to compute the conformal quantile.
4. Call `.predict_intervals(y_hat_new)` for new predictions.

Note on this project's weak-supervision setting
------------------------------------------------
Training labels are zone-level averages (only 3 distinct values).
Conformal residuals |y_zone − ŷ_cell| therefore capture how well the model
reproduces zone-level targets, not fine-grained ground truth.
Intervals have valid marginal coverage over the calibration distribution
but should be interpreted as uncertainty over spatial allocation, not
as bounds on true individual-cell deprivation.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SplitConformalPredictor:
    """
    Model-agnostic split-conformal prediction interval calibrator.

    Works with any model's point predictions — fit the model separately,
    then use this class to calibrate and produce intervals.

    Parameters
    ----------
    coverage : float
        Target marginal coverage level (default 0.90 → 90% CI).
    """

    def __init__(self, coverage: float = 0.90):
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}.")
        self.coverage = coverage
        self.q_hat: float | None = None
        self._n_cal: int = 0

    def calibrate(
        self,
        y_cal: np.ndarray,
        y_hat_cal: np.ndarray,
    ) -> "SplitConformalPredictor":
        """
        Compute the conformal quantile from calibration nonconformity scores.

        Nonconformity score: absolute residual |y − ŷ|.
        Quantile level uses the finite-sample correction
            ⌈(1 − α)(n + 1)⌉ / n
        so coverage is guaranteed for any n ≥ 1.

        Parameters
        ----------
        y_cal : np.ndarray, shape (n_cal,)
            True labels on the calibration set.
        y_hat_cal : np.ndarray, shape (n_cal,)
            Model point predictions on the calibration set.

        Returns
        -------
        self
        """
        y_cal = np.asarray(y_cal, dtype=float)
        y_hat_cal = np.asarray(y_hat_cal, dtype=float)

        mask = ~(np.isnan(y_cal) | np.isnan(y_hat_cal))
        scores = np.abs(y_cal[mask] - y_hat_cal[mask])
        n = len(scores)

        if n < 5:
            logger.warning(
                "Only %d calibration samples — conformal quantile may be unreliable.", n
            )

        alpha = 1.0 - self.coverage
        # Finite-sample corrected level (clipped to 1.0 for safety)
        level = min(np.ceil((1.0 - alpha) * (n + 1)) / n, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        self._n_cal = n

        logger.info(
            "Conformal calibration: n_cal=%d, coverage=%.0f%%, q̂=%.4f",
            n, self.coverage * 100, self.q_hat,
        )
        return self

    def predict_intervals(
        self,
        y_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Produce symmetric conformal prediction intervals.

        Parameters
        ----------
        y_hat : np.ndarray
            Point predictions for new data.

        Returns
        -------
        (lower, upper) : (np.ndarray, np.ndarray)
            Prediction interval bounds.
        """
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate() before .predict_intervals().")
        y_hat = np.asarray(y_hat, dtype=float)
        return y_hat - self.q_hat, y_hat + self.q_hat

    @property
    def interval_width(self) -> float:
        """Total width of the symmetric interval (2 × q̂)."""
        return 2.0 * self.q_hat if self.q_hat is not None else float("nan")


def calibration_split(
    X: np.ndarray,
    y: np.ndarray,
    cal_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split arrays into training and calibration subsets.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
    y : np.ndarray, shape (n,)
    cal_fraction : float
        Fraction of samples reserved for calibration.
    random_state : int

    Returns
    -------
    X_train, X_cal, y_train, y_cal
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    n_cal = max(1, int(n * cal_fraction))
    cal_idx = rng.choice(n, size=n_cal, replace=False)
    train_idx = np.setdiff1d(np.arange(n), cal_idx)
    return X[train_idx], X[cal_idx], y[train_idx], y[cal_idx]
