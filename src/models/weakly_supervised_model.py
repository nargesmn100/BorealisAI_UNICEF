"""
Weakly Supervised Spatial Disaggregation Model
===============================================

Scientific framing
------------------
The key problem: we have fine-scale features for every grid cell (~2.3 km),
but poverty ground truth is only available at the coarse zone level (3 zones:
Urban / Rural / KMA).

The principled solution (from the new problem statement) is to train the model
with a REGION-LEVEL loss, not a cell-level loss:

    L = Σ_r ( Ŷ_r - Y_r )²

where the zone prediction is a population-weighted aggregate of cell predictions:

    Ŷ_r = Σ_{i∈r} p_i · ŷ_i  /  Σ_{i∈r} p_i

and Y_r is the official zone-level poverty prevalence (the only ground truth
we have).

Why this differs from the existing Ridge / GBM approach
-------------------------------------------------------
The existing models assign each cell the zone-level average as its "label"
and train a cell-level MSE loss. Spatial variation then comes entirely from
post-hoc reconciliation (rescaling raw scores to match zone totals).

This model instead:
  1. Predicts a score for each cell
  2. Aggregates predictions within each zone using population weights
  3. Computes the loss at the zone level
  4. Updates model weights to reduce zone-level error

The spatial variation the model learns comes from the features themselves,
guided by the zone-level constraint — the model discovers which feature
patterns explain the coarse-scale totals.

Two model classes
-----------------
WeaklySupervisedLinear
    Linear model: ŷ_i = X_i @ w + b

    For linear models, the population-weighted zone aggregate is:
        Ŷ_r = X̄_r @ w + b
    where X̄_r is the pop-weighted mean feature vector for zone r.

    So the zone-level loss reduces to linear regression on 3 zone-level
    "super-rows" — but the model is applied at cell level for prediction.
    We use L-BFGS-B via scipy.optimize for correctness and to allow
    the same training loop as the MLP.

WeaklySupervisedMLP
    Small neural network: ŷ_i = W2 · tanh(W1 · x_i + b1) + b2

    The zone-level aggregation is non-linear in parameters, so
    we compute gradients analytically via backpropagation through
    the aggregation operation and optimize with L-BFGS-B.

    Architecture: input(9) → Linear → tanh → Linear(1)

Region-based evaluation
-----------------------
With only 3 zones, the leave-one-zone-out CV tests whether the model
can generalise to a zone it has never seen during training.
See src/evaluation/region_split.py for the evaluation harness.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from src.reconciliation.admin_reconcile import reconcile_predictions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared aggregation utilities
# ---------------------------------------------------------------------------

def _population_weighted_zone_predictions(
    cell_preds: np.ndarray,
    zone_labels: np.ndarray,
    populations: np.ndarray,
) -> dict:
    """
    Aggregate cell-level predictions into zone-level predictions using
    population weighting.

    Parameters
    ----------
    cell_preds : np.ndarray, shape (n,)
    zone_labels : np.ndarray of str/int, shape (n,)
    populations : np.ndarray, shape (n,)

    Returns
    -------
    dict : {zone_label: predicted_zone_prevalence}
    """
    zones = np.unique(zone_labels)
    zone_preds = {}
    for zone in zones:
        mask = zone_labels == zone
        pop = populations[mask].astype(float)
        pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)
        preds = cell_preds[mask]
        if pop.sum() > 0:
            zone_preds[zone] = np.average(preds, weights=pop)
        else:
            zone_preds[zone] = preds.mean()
    return zone_preds


def _zone_level_mse_loss(
    cell_preds: np.ndarray,
    zone_labels: np.ndarray,
    populations: np.ndarray,
    zone_targets: dict,
    l2_reg: float = 0.0,
    params: Optional[np.ndarray] = None,
) -> float:
    """
    Compute zone-level MSE loss (optionally with L2 regularisation).

    Loss = Σ_r (Ŷ_r − Y_r)²  +  λ · ‖params‖²

    Parameters
    ----------
    cell_preds : predicted scores for each cell
    zone_labels : zone assignment for each cell
    populations : population count for each cell
    zone_targets : {zone: official_target_value}
    l2_reg : L2 regularisation coefficient
    params : model parameters (for L2 penalty); None disables L2.
    """
    zone_preds = _population_weighted_zone_predictions(
        cell_preds, zone_labels, populations
    )
    loss = 0.0
    for zone, target in zone_targets.items():
        if zone in zone_preds:
            loss += (zone_preds[zone] - target) ** 2
    if l2_reg > 0 and params is not None:
        loss += l2_reg * np.dot(params, params)
    return loss


# ---------------------------------------------------------------------------
# WeaklySupervisedLinear
# ---------------------------------------------------------------------------

class WeaklySupervisedLinear:
    """
    Linear model trained with population-weighted zone-level MSE loss.

    ŷ_i = X_i @ w + b

    Zone prediction: Ŷ_r = X̄_r @ w + b  (pop-weighted mean features)

    Loss: Σ_r (Ŷ_r − Y_r)²  +  λ‖w‖²

    Because this reduces to linear regression on zone-level aggregated
    features, the solution is unique and fast. We still use scipy.optimize
    so the training loop is transparent and matches the MLP.

    Parameters
    ----------
    l2_reg : float
        L2 regularisation on weights (not bias). Prevents overfitting
        when the number of zones is much smaller than the number of features.
    random_state : int
    """

    def __init__(self, l2_reg: float = 1.0, random_state: int = 42):
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.w_: Optional[np.ndarray] = None
        self.b_: Optional[float] = None
        self.feature_names_: list = []
        self.train_loss_: Optional[float] = None
        self.optimisation_result_ = None

    def _params_to_wb(self, params: np.ndarray):
        return params[:-1], params[-1]

    def _loss_and_grad(
        self,
        params: np.ndarray,
        X_scaled: np.ndarray,
        zone_labels: np.ndarray,
        populations: np.ndarray,
        zone_targets: dict,
    ):
        """Compute zone-level MSE loss and gradient w.r.t. params."""
        w, b = self._params_to_wb(params)
        y_hat = X_scaled @ w + b  # (n,)

        # Zone-level predictions and residuals
        zones = list(zone_targets.keys())
        loss = 0.0
        # Gradient of loss w.r.t. each cell prediction
        # dL/dŷ_i = 2(Ŷ_{r(i)} − Y_{r(i)}) · p_i / Σ_{j∈r(i)} p_j
        dL_dy = np.zeros(len(y_hat))

        for zone in zones:
            target = zone_targets[zone]
            mask = zone_labels == zone
            pop = populations[mask].astype(float)
            pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)
            total_pop = pop.sum()

            if total_pop > 0:
                y_zone = np.average(y_hat[mask], weights=pop)
            else:
                y_zone = y_hat[mask].mean()
                total_pop = mask.sum()

            residual = y_zone - target
            loss += residual ** 2

            # Gradient of zone loss w.r.t. cell predictions in this zone
            if total_pop > 0:
                dL_dy[mask] += 2.0 * residual * pop / total_pop
            else:
                dL_dy[mask] += 2.0 * residual / mask.sum()

        # L2 regularisation on w (not bias)
        loss += self.l2_reg * np.dot(w, w)

        # Gradient: dL/dw = X.T @ dL_dy + 2λw
        dL_dw = X_scaled.T @ dL_dy + 2.0 * self.l2_reg * w
        dL_db = dL_dy.sum()

        grad = np.append(dL_dw, dL_db)
        return float(loss), grad

    def fit(
        self,
        X: pd.DataFrame,
        zone_labels: np.ndarray,
        populations: np.ndarray,
        zone_targets: dict,
        feature_names: Optional[list] = None,
    ) -> "WeaklySupervisedLinear":
        """
        Fit the model by minimising zone-level MSE loss.

        Parameters
        ----------
        X : feature matrix (n_cells × n_features)
        zone_labels : zone assignment for each cell
        populations : population per cell
        zone_targets : {zone_name: official_poverty_target}
        feature_names : column names for importance reporting
        """
        if feature_names is not None:
            self.feature_names_ = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.fit_transform(X_arr)

        n_features = X_scaled.shape[1]
        rng = np.random.default_rng(self.random_state)
        params0 = rng.normal(0, 0.01, n_features + 1)

        logger.info(
            "WeaklySupervisedLinear: optimising zone-level MSE loss "
            "(n_cells=%d, n_zones=%d, l2=%.4f)...",
            len(X_scaled), len(zone_targets), self.l2_reg,
        )

        result = minimize(
            fun=self._loss_and_grad,
            x0=params0,
            args=(X_scaled, zone_labels, populations, zone_targets),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

        self.optimisation_result_ = result
        self.w_, self.b_ = self._params_to_wb(result.x)
        self.train_loss_ = result.fun

        logger.info(
            "WeaklySupervisedLinear converged: success=%s, "
            "final_loss=%.6f, n_iter=%d",
            result.success, result.fun, result.nit,
        )

        # Log achieved zone predictions
        y_hat = X_scaled @ self.w_ + self.b_
        zone_preds = _population_weighted_zone_predictions(
            y_hat, zone_labels, populations
        )
        logger.info("Zone-level aggregation check (raw, before reconciliation):")
        for zone, target in zone_targets.items():
            pred = zone_preds.get(zone, float("nan"))
            logger.info(
                "  %-50s | target=%.3f | raw_pred=%.3f | diff=%.4f",
                zone, target, pred, abs(pred - target),
            )

        # Log feature importances (standardised coefficients)
        self._log_feature_importances()

        return self

    def _log_feature_importances(self):
        """Log feature importances as standardised coefficients."""
        if self.w_ is None:
            return
        # Since input is StandardScaler-normalised, coef magnitudes are comparable
        importances = np.abs(self.w_)
        sorted_idx = np.argsort(importances)[::-1]
        logger.info("WeaklySupervisedLinear feature importances (|coefficient|):")
        for i in sorted_idx:
            name = (self.feature_names_[i] if i < len(self.feature_names_)
                    else f"feat_{i}")
            logger.info("  %-35s : %+.4f", name, self.w_[i])

    def predict(self, X) -> np.ndarray:
        """Predict cell-level deprivation scores."""
        if self.w_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)
        return X_scaled @ self.w_ + self.b_

    def feature_importance_permutation(
        self,
        X: np.ndarray,
        zone_labels: np.ndarray,
        populations: np.ndarray,
        zone_targets: dict,
        n_repeats: int = 30,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Compute permutation feature importance.

        For each feature, shuffle its values across all cells, re-predict,
        re-aggregate, and measure how much the zone-level loss increases.
        A large increase = that feature is important.

        Parameters
        ----------
        X : scaled feature matrix (output of scaler.transform)
        zone_labels, populations, zone_targets : same as fit()
        n_repeats : number of shuffle repeats per feature
        random_state : seed

        Returns
        -------
        pd.DataFrame with columns: feature, importance_mean, importance_std
        """
        if self.w_ is None:
            raise RuntimeError("Model not fitted.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)

        rng = np.random.default_rng(random_state)
        base_preds = X_scaled @ self.w_ + self.b_
        base_loss = _zone_level_mse_loss(
            base_preds, zone_labels, populations, zone_targets
        )

        results = []
        n_features = X_scaled.shape[1]

        for feat_idx in range(n_features):
            delta_losses = []
            for _ in range(n_repeats):
                X_perm = X_scaled.copy()
                perm_idx = rng.permutation(len(X_perm))
                X_perm[:, feat_idx] = X_perm[perm_idx, feat_idx]
                perm_preds = X_perm @ self.w_ + self.b_
                perm_loss = _zone_level_mse_loss(
                    perm_preds, zone_labels, populations, zone_targets
                )
                delta_losses.append(perm_loss - base_loss)

            name = (self.feature_names_[feat_idx]
                    if feat_idx < len(self.feature_names_) else f"feat_{feat_idx}")
            results.append({
                "feature": name,
                "importance_mean": float(np.mean(delta_losses)),
                "importance_std": float(np.std(delta_losses)),
            })

        df_imp = pd.DataFrame(results).sort_values(
            "importance_mean", ascending=False
        ).reset_index(drop=True)

        logger.info("Permutation feature importances (zone-level loss increase):")
        for _, row in df_imp.iterrows():
            logger.info("  %-35s : %.6f ± %.6f",
                        row["feature"], row["importance_mean"], row["importance_std"])

        return df_imp


# ---------------------------------------------------------------------------
# WeaklySupervisedMLP
# ---------------------------------------------------------------------------

class WeaklySupervisedMLP:
    """
    Small neural network trained with population-weighted zone-level MSE loss.

    Architecture: Input → Linear(hidden_size) → tanh → Linear(1)

    ŷ_i = W2 · tanh(W1 · x_i + b1) + b2

    Zone prediction: Ŷ_r = Σ_{i∈r} p_i · ŷ_i / Σ_{i∈r} p_i

    Loss: Σ_r (Ŷ_r − Y_r)²  +  λ(‖W1‖² + ‖W2‖²)

    The gradient flows back through the aggregation operation analytically.
    We do NOT use any deep learning framework — only numpy + scipy.

    Why an MLP here:
    - The linear model can only separate zones through their mean features
    - The MLP can learn non-linear feature interactions within zones
      (e.g. high RWI + high child population behaves differently than
       high RWI + low child population in urban vs rural settings)
    - With only 3 zone-level supervision signals this is still highly
      regularised — the architecture is small intentionally

    Parameters
    ----------
    hidden_size : int
        Number of hidden units. Keep small (8–32) given limited supervision.
    l2_reg : float
        L2 regularisation on all weight matrices (not biases).
    max_iter : int
        Max optimiser iterations.
    random_state : int
    """

    def __init__(
        self,
        hidden_size: int = 16,
        l2_reg: float = 0.01,
        max_iter: int = 2000,
        random_state: int = 42,
    ):
        self.hidden_size = hidden_size
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.params_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.feature_names_: list = []
        self.train_loss_: Optional[float] = None
        self.optimisation_result_ = None

    # ------------------------------------------------------------------
    # Parameter packing / unpacking
    # ------------------------------------------------------------------

    def _param_shapes(self, n_features: int):
        """Return shapes for W1, b1, W2, b2."""
        return (
            (n_features, self.hidden_size),  # W1
            (self.hidden_size,),              # b1
            (self.hidden_size, 1),            # W2
            (1,),                             # b2
        )

    def _n_params(self, n_features: int) -> int:
        shapes = self._param_shapes(n_features)
        return sum(np.prod(s) for s in shapes)

    def _unpack(self, params: np.ndarray, n_features: int):
        shapes = self._param_shapes(n_features)
        sizes = [int(np.prod(s)) for s in shapes]
        splits = np.cumsum(sizes[:-1])
        parts = np.split(params, splits)
        W1 = parts[0].reshape(shapes[0])
        b1 = parts[1].reshape(shapes[1])
        W2 = parts[2].reshape(shapes[2])
        b2 = parts[3].reshape(shapes[3])
        return W1, b1, W2, b2

    def _pack(self, W1, b1, W2, b2) -> np.ndarray:
        return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, X_scaled: np.ndarray, params: np.ndarray):
        """
        Forward pass.

        Returns
        -------
        y_hat : (n,)  — cell-level predictions
        h     : (n, hidden_size) — pre-activation hidden states (for backprop)
        a     : (n, hidden_size) — post-activation hidden states (for backprop)
        """
        W1, b1, W2, b2 = self._unpack(params, self.n_features_)
        h = X_scaled @ W1 + b1          # (n, hidden)
        a = np.tanh(h)                   # (n, hidden)
        y_hat = (a @ W2 + b2).ravel()   # (n,)
        return y_hat, h, a

    # ------------------------------------------------------------------
    # Loss + gradient
    # ------------------------------------------------------------------

    def _loss_and_grad(
        self,
        params: np.ndarray,
        X_scaled: np.ndarray,
        zone_labels: np.ndarray,
        populations: np.ndarray,
        zone_targets: dict,
    ):
        """
        Compute zone-level MSE loss and gradient w.r.t. all parameters.

        Backpropagation through the population-weighted aggregation:

            Ŷ_r = Σ_{i∈r} p_i · ŷ_i / Σ_{i∈r} p_i

            dL/dŷ_i = 2(Ŷ_{r(i)} − Y_{r(i)}) · p_i / Σ_{j∈r(i)} p_j
        """
        W1, b1, W2, b2 = self._unpack(params, self.n_features_)
        y_hat, h, a = self._forward(X_scaled, params)

        # 1. Zone-level aggregation and loss
        loss = 0.0
        dL_dy = np.zeros(len(y_hat))

        for zone, target in zone_targets.items():
            mask = zone_labels == zone
            pop = populations[mask].astype(float)
            pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)
            total_pop = pop.sum()

            if total_pop > 0:
                y_zone = np.average(y_hat[mask], weights=pop)
                residual = y_zone - target
                loss += residual ** 2
                # Gradient w.r.t. cell preds in this zone
                dL_dy[mask] += 2.0 * residual * pop / total_pop
            else:
                n_zone = mask.sum()
                if n_zone > 0:
                    y_zone = y_hat[mask].mean()
                    residual = y_zone - target
                    loss += residual ** 2
                    dL_dy[mask] += 2.0 * residual / n_zone

        # 2. L2 regularisation (weights only, not biases)
        loss += self.l2_reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

        # 3. Backprop through output layer: y = a @ W2 + b2
        #    dL/da = dL/dy · W2.T      (n, hidden)
        dL_dy_col = dL_dy.reshape(-1, 1)         # (n, 1)
        dL_dW2 = a.T @ dL_dy_col                 # (hidden, 1)
        dL_db2 = dL_dy.sum(keepdims=True)         # (1,)

        # 4. Backprop through hidden layer: h = X @ W1 + b1, a = tanh(h)
        dL_da = dL_dy_col @ W2.T                  # (n, hidden)
        dL_dh = dL_da * (1.0 - a ** 2)           # tanh derivative: (n, hidden)
        dL_dW1 = X_scaled.T @ dL_dh               # (n_features, hidden)
        dL_db1 = dL_dh.sum(axis=0)               # (hidden,)

        # 5. L2 gradient
        dL_dW1 += 2.0 * self.l2_reg * W1
        dL_dW2 += 2.0 * self.l2_reg * W2

        grad = self._pack(dL_dW1, dL_db1, dL_dW2, dL_db2)
        return float(loss), grad

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        zone_labels: np.ndarray,
        populations: np.ndarray,
        zone_targets: dict,
        feature_names: Optional[list] = None,
    ) -> "WeaklySupervisedMLP":
        """
        Fit MLP by minimising zone-level MSE loss via L-BFGS-B.

        Parameters
        ----------
        X : feature matrix
        zone_labels : zone assignment per cell
        populations : population per cell
        zone_targets : {zone_name: official_target}
        feature_names : column names for logging
        """
        if feature_names is not None:
            self.feature_names_ = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.fit_transform(X_arr)
        self.n_features_ = X_scaled.shape[1]

        rng = np.random.default_rng(self.random_state)
        # He initialisation for tanh network
        std = np.sqrt(2.0 / self.n_features_)
        W1_init = rng.normal(0, std, (self.n_features_, self.hidden_size))
        b1_init = np.zeros(self.hidden_size)
        W2_init = rng.normal(0, 0.1, (self.hidden_size, 1))
        b2_init = np.zeros(1)
        params0 = self._pack(W1_init, b1_init, W2_init, b2_init)

        logger.info(
            "WeaklySupervisedMLP: optimising zone-level MSE loss "
            "(n_cells=%d, n_zones=%d, hidden=%d, l2=%.4f)...",
            len(X_scaled), len(zone_targets), self.hidden_size, self.l2_reg,
        )

        result = minimize(
            fun=self._loss_and_grad,
            x0=params0,
            args=(X_scaled, zone_labels, populations, zone_targets),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-12, "gtol": 1e-8},
        )

        self.optimisation_result_ = result
        self.params_ = result.x
        self.train_loss_ = result.fun

        logger.info(
            "WeaklySupervisedMLP converged: success=%s, "
            "final_loss=%.6f, n_iter=%d",
            result.success, result.fun, result.nit,
        )

        # Log zone-level aggregation accuracy
        y_hat, _, _ = self._forward(X_scaled, self.params_)
        zone_preds = _population_weighted_zone_predictions(
            y_hat, zone_labels, populations
        )
        logger.info("Zone-level aggregation check (raw, before reconciliation):")
        for zone, target in zone_targets.items():
            pred = zone_preds.get(zone, float("nan"))
            logger.info(
                "  %-50s | target=%.3f | raw_pred=%.3f | diff=%.4f",
                zone, target, pred, abs(pred - target),
            )

        return self

    def predict(self, X) -> np.ndarray:
        """Predict cell-level scores."""
        if self.params_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)
        y_hat, _, _ = self._forward(X_scaled, self.params_)
        return y_hat

    def feature_importance_permutation(
        self,
        X,
        zone_labels: np.ndarray,
        populations: np.ndarray,
        zone_targets: dict,
        n_repeats: int = 30,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Permutation feature importance measured by zone-level loss increase.

        See WeaklySupervisedLinear.feature_importance_permutation for details.
        """
        if self.params_ is None:
            raise RuntimeError("Model not fitted.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)

        rng = np.random.default_rng(random_state)
        base_preds = self.predict(X)
        base_loss = _zone_level_mse_loss(
            base_preds, zone_labels, populations, zone_targets
        )

        results = []
        n_features = X_scaled.shape[1]

        for feat_idx in range(n_features):
            delta_losses = []
            for _ in range(n_repeats):
                X_perm_scaled = X_scaled.copy()
                perm_idx = rng.permutation(len(X_perm_scaled))
                X_perm_scaled[:, feat_idx] = X_perm_scaled[perm_idx, feat_idx]
                # Predict with permuted scaled features directly
                y_hat_perm, _, _ = self._forward(X_perm_scaled, self.params_)
                perm_loss = _zone_level_mse_loss(
                    y_hat_perm, zone_labels, populations, zone_targets
                )
                delta_losses.append(perm_loss - base_loss)

            name = (self.feature_names_[feat_idx]
                    if feat_idx < len(self.feature_names_) else f"feat_{feat_idx}")
            results.append({
                "feature": name,
                "importance_mean": float(np.mean(delta_losses)),
                "importance_std": float(np.std(delta_losses)),
            })

        df_imp = pd.DataFrame(results).sort_values(
            "importance_mean", ascending=False
        ).reset_index(drop=True)

        logger.info("MLP permutation feature importances (zone-level loss increase):")
        for _, row in df_imp.iterrows():
            logger.info("  %-35s : %.6f ± %.6f",
                        row["feature"], row["importance_mean"], row["importance_std"])

        return df_imp


# ---------------------------------------------------------------------------
# Pipeline run() function
# ---------------------------------------------------------------------------

def run(cfg: dict, df: pd.DataFrame) -> tuple:
    """
    Fit both WeaklySupervisedLinear and WeaklySupervisedMLP, generate
    predictions, reconcile to official zone totals, compute feature importances.

    Parameters
    ----------
    cfg : dict — loaded config
    df : pd.DataFrame — modeling table from data pipeline

    Returns
    -------
    (df, ws_linear_model, ws_mlp_model, linear_imp_df, mlp_imp_df)
    """
    feature_cols = cfg["modeling"]["features"]
    target_moderate = "moderate_prevalence"
    target_severe = "severe_prevalence"
    zone_col = cfg["modeling"]["admin_zone_col"]
    pop_col = "population"
    ws_cfg = cfg["modeling"].get("weakly_supervised", {})

    # ------------------------------------------------------------------
    # Prepare training data
    # Only cells that (a) are in the modeling sample, (b) have complete features
    # ------------------------------------------------------------------
    model_mask = df["in_modeling_sample"].fillna(False)
    feature_mask = df[feature_cols].notna().all(axis=1)
    train_mask = model_mask & feature_mask

    logger.info(
        "WeaklySupervisedModel: %d training cells with complete features "
        "out of %d in modeling sample.",
        train_mask.sum(), model_mask.sum(),
    )

    if train_mask.sum() < 10:
        raise ValueError(
            f"Only {train_mask.sum()} training samples. Cannot fit model."
        )

    X_train = df.loc[train_mask, feature_cols]
    zone_labels_train = df.loc[train_mask, zone_col].values
    pop_train = df.loc[train_mask, pop_col].values.astype(float)

    # Build zone_targets dict from the training data
    # (each zone has a single official target, taken from first row)
    zone_targets_moderate = (
        df.loc[train_mask]
        .groupby(zone_col)[target_moderate]
        .first()
        .to_dict()
    )
    zone_targets_severe = (
        df.loc[train_mask]
        .groupby(zone_col)[target_severe]
        .first()
        .to_dict()
    )

    logger.info("Zone targets (moderate): %s", zone_targets_moderate)
    logger.info("Zone targets (severe):   %s", zone_targets_severe)

    # Prediction mask: all cells with valid features and known zone
    pred_mask = feature_mask & (df[zone_col].notna()) & (df[zone_col] != "Unknown")
    X_pred = df.loc[pred_mask, feature_cols]

    # ------------------------------------------------------------------
    # --- WeaklySupervisedLinear ---
    # ------------------------------------------------------------------
    linear_cfg = ws_cfg.get("linear", {})
    l2_reg_linear = linear_cfg.get("l2_reg", 1.0)
    rs_linear = linear_cfg.get("random_state", 42)

    logger.info("--- Fitting WeaklySupervisedLinear (moderate) ---")
    ws_linear = WeaklySupervisedLinear(l2_reg=l2_reg_linear, random_state=rs_linear)
    ws_linear.fit(
        X_train, zone_labels_train, pop_train,
        zone_targets_moderate, feature_names=feature_cols,
    )

    raw_linear = ws_linear.predict(X_pred)
    df = df.copy()
    df["ws_linear_raw"] = np.nan
    df.loc[pred_mask, "ws_linear_raw"] = raw_linear

    # Reconcile moderate
    logger.info("Reconciling ws_linear predictions → moderate...")
    df = reconcile_predictions(
        df, "ws_linear_raw", target_moderate, zone_col, pop_col,
        output_col="ws_linear_moderate", strategy="population_weighted",
    )

    # Reconcile severe: fit a second model on severe targets
    logger.info("--- Fitting WeaklySupervisedLinear (severe) ---")
    ws_linear_sev = WeaklySupervisedLinear(l2_reg=l2_reg_linear, random_state=rs_linear)
    ws_linear_sev.fit(
        X_train, zone_labels_train, pop_train,
        zone_targets_severe, feature_names=feature_cols,
    )
    raw_linear_sev = ws_linear_sev.predict(X_pred)
    df["ws_linear_raw_severe"] = np.nan
    df.loc[pred_mask, "ws_linear_raw_severe"] = raw_linear_sev
    df = reconcile_predictions(
        df, "ws_linear_raw_severe", target_severe, zone_col, pop_col,
        output_col="ws_linear_severe", strategy="population_weighted",
    )

    # Permutation importance (moderate model)
    logger.info("Computing permutation feature importances for ws_linear...")
    linear_imp_df = ws_linear.feature_importance_permutation(
        X_train, zone_labels_train, pop_train, zone_targets_moderate,
        n_repeats=ws_cfg.get("n_importance_repeats", 30),
        random_state=rs_linear,
    )

    # ------------------------------------------------------------------
    # --- WeaklySupervisedMLP ---
    # ------------------------------------------------------------------
    mlp_cfg = ws_cfg.get("mlp", {})
    hidden_size = mlp_cfg.get("hidden_size", 16)
    l2_reg_mlp = mlp_cfg.get("l2_reg", 0.01)
    max_iter = mlp_cfg.get("max_iter", 2000)
    rs_mlp = mlp_cfg.get("random_state", 42)

    logger.info("--- Fitting WeaklySupervisedMLP (moderate) ---")
    ws_mlp = WeaklySupervisedMLP(
        hidden_size=hidden_size, l2_reg=l2_reg_mlp,
        max_iter=max_iter, random_state=rs_mlp,
    )
    ws_mlp.fit(
        X_train, zone_labels_train, pop_train,
        zone_targets_moderate, feature_names=feature_cols,
    )

    raw_mlp = ws_mlp.predict(X_pred)
    df["ws_mlp_raw"] = np.nan
    df.loc[pred_mask, "ws_mlp_raw"] = raw_mlp

    # Reconcile moderate
    logger.info("Reconciling ws_mlp predictions → moderate...")
    df = reconcile_predictions(
        df, "ws_mlp_raw", target_moderate, zone_col, pop_col,
        output_col="ws_mlp_moderate", strategy="population_weighted",
    )

    # Reconcile severe
    logger.info("--- Fitting WeaklySupervisedMLP (severe) ---")
    ws_mlp_sev = WeaklySupervisedMLP(
        hidden_size=hidden_size, l2_reg=l2_reg_mlp,
        max_iter=max_iter, random_state=rs_mlp,
    )
    ws_mlp_sev.fit(
        X_train, zone_labels_train, pop_train,
        zone_targets_severe, feature_names=feature_cols,
    )
    raw_mlp_sev = ws_mlp_sev.predict(X_pred)
    df["ws_mlp_raw_severe"] = np.nan
    df.loc[pred_mask, "ws_mlp_raw_severe"] = raw_mlp_sev
    df = reconcile_predictions(
        df, "ws_mlp_raw_severe", target_severe, zone_col, pop_col,
        output_col="ws_mlp_severe", strategy="population_weighted",
    )

    # Permutation importance (MLP moderate model)
    logger.info("Computing permutation feature importances for ws_mlp...")
    mlp_imp_df = ws_mlp.feature_importance_permutation(
        X_train, zone_labels_train, pop_train, zone_targets_moderate,
        n_repeats=ws_cfg.get("n_importance_repeats", 30),
        random_state=rs_mlp,
    )

    logger.info("Weakly supervised models complete.")
    return df, ws_linear, ws_mlp, linear_imp_df, mlp_imp_df
