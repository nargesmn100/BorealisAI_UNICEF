"""
Weakly Supervised Neural Network with Aggregation-Based Training
=================================================================

Core innovation: Train via region-level aggregation loss, not cell-level labels.

The model learns because predictions must aggregate correctly, not by being
forced to aggregate correctly afterward (post-processing reconciliation).

Training procedure:
    1. Predict prevalence for each cell: ŷ_i = f(x_i)
    2. Aggregate predictions within each supervision group (population-weighted):
       Ŷ_g = Σ(ŷ_i × p_i) / Σ(p_i)
    3. Compare aggregates to official group targets: L = Σ_g (Ŷ_g − Y_g)²
    4. Backpropagate through aggregation to update model

This implements the core methodology from the problem statement:
"weakly supervised geospatial learning with aggregation-based training."
"""

import logging
import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Weakly supervised NN will not be available.")


class WeaklySupervisedNN:
    """
    Neural network trained via region-level aggregation loss.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dims : list of int, default=[64, 32]
        Hidden layer dimensions
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer
    dropout : float, default=0.2
        Dropout probability for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32],
        learning_rate: float = 0.001,
        dropout: float = 0.2
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for WeaklySupervisedNN. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.dropout = dropout
        
        # Build model
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Track training history
        self.loss_history = []
        
    def _build_model(self) -> nn.Module:
        """
        Build feedforward neural network.
        
        Architecture:
            Input → [Linear → ReLU → Dropout] × layers → Linear → Sigmoid → Output
        
        Sigmoid constrains output to [0, 1] (prevalence as probability).
        """
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Constrain to [0, 1]
        
        return nn.Sequential(*layers)
    
    def compute_aggregation_loss(
        self,
        cell_predictions: torch.Tensor,
        cell_populations: torch.Tensor,
        supervision_groups: torch.Tensor,
        group_targets: Dict[int, float]
    ) -> torch.Tensor:
        """
        Core innovation: Compute loss on population-weighted aggregates.
        
        Parameters
        ----------
        cell_predictions : torch.Tensor, shape [n_cells, 1]
            Predicted prevalence per cell (0-1 scale)
        cell_populations : torch.Tensor, shape [n_cells]
            Population weight per cell
        supervision_groups : torch.Tensor, shape [n_cells]
            Group ID per cell (0-indexed)
        group_targets : dict
            {group_id: target_prevalence} mapping (0-1 scale)
        
        Returns
        -------
        loss : torch.Tensor
            Mean squared error on aggregated predictions
        """
        losses = []

        for group_id in torch.unique(supervision_groups):
            # Select cells in this group
            mask = (supervision_groups == group_id)
            group_preds = cell_predictions[mask].squeeze()
            group_pops = cell_populations[mask]

            # Skip if group has no cells
            if len(group_preds) == 0:
                continue

            # Population-weighted aggregate: Σ(pred_i × pop_i) / Σ(pop_i)
            agg_pred = (group_preds * group_pops).sum() / group_pops.sum()

            # Get official target for this group
            target = group_targets.get(group_id.item())
            if target is None:
                logger.warning(f"No target found for group {group_id.item()}, skipping")
                continue

            target_tensor = torch.tensor(target, dtype=torch.float32)

            # Squared error on aggregate
            losses.append((agg_pred - target_tensor) ** 2)

        # Mean over groups
        if len(losses) > 0:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        return loss
    
    def fit(
        self,
        X: np.ndarray,
        population: np.ndarray,
        supervision_groups: np.ndarray,
        group_targets: Dict[int, float],
        n_epochs: int = 100,
        verbose: bool = True
    ) -> 'WeaklySupervisedNN':
        """
        Train with aggregation-based loss.
        
        Parameters
        ----------
        X : np.ndarray, shape [n_cells, n_features]
            Feature matrix
        population : np.ndarray, shape [n_cells]
            Population per cell (for weighting aggregation)
        supervision_groups : np.ndarray, shape [n_cells]
            Group ID per cell (integer 0-indexed)
        group_targets : dict
            {group_id: target_prevalence} (0-1 scale, e.g., 0.3465 for 34.65%)
        n_epochs : int, default=100
            Number of training epochs
        verbose : bool, default=True
            Print training progress
        
        Returns
        -------
        self : WeaklySupervisedNN
            Fitted model
        """
        # Convert to PyTorch tensors (copy to ensure writable arrays from Parquet)
        X_t = torch.FloatTensor(np.array(X, copy=True))
        pop_t = torch.FloatTensor(np.array(population, copy=True, dtype=np.float32))
        groups_t = torch.LongTensor(np.array(supervision_groups, copy=True))
        
        # Normalize features (important for neural networks)
        self.feature_mean = X_t.mean(dim=0, keepdim=True)
        self.feature_std = X_t.std(dim=0, keepdim=True) + 1e-8
        X_normalized = (X_t - self.feature_mean) / self.feature_std
        
        # Training loop
        self.model.train()
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X_normalized)
            
            # Compute aggregation loss
            loss = self.compute_aggregation_loss(
                predictions, pop_t, groups_t, group_targets
            )
            
            # Backprop and update
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            self.loss_history.append(loss.item())
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                logger.info(f"  Epoch {epoch:3d}/{n_epochs} | Loss: {loss.item():.6f}")

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cell-level prevalence.
        
        Parameters
        ----------
        X : np.ndarray, shape [n_cells, n_features]
            Feature matrix
        
        Returns
        -------
        predictions : np.ndarray, shape [n_cells]
            Predicted prevalence (0-100 percentage scale)
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(np.array(X, copy=True))
            X_normalized = (X_t - self.feature_mean) / self.feature_std
            preds = self.model(X_normalized).numpy()
        
        # Convert from [0, 1] to [0, 100] percentage
        return preds.squeeze() * 100


def _prepare_zone_groups(df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Prepare zone-level supervision groups (dynamically from subregion column).

    Parameters
    ----------
    df : pd.DataFrame
        Modeling table with 'subregion' column
    target_col : str
        Target column name (e.g., 'moderate_prevalence')

    Returns
    -------
    groups : np.ndarray, shape [n_cells]
        Integer group IDs
    targets : dict
        {group_id: target_prevalence} (0-1 scale)
    """
    # Dynamically map zone names to integer IDs
    zone_names = sorted(df['subregion'].dropna().unique())
    zone_map = {name: idx for idx, name in enumerate(zone_names)}

    groups = df['subregion'].map(zone_map).values

    # Extract targets for each zone
    targets = {}
    for zone_name, zone_id in zone_map.items():
        zone_mask = df['subregion'] == zone_name
        if zone_mask.sum() > 0:
            target_val = df.loc[zone_mask, target_col].iloc[0] / 100  # Convert % to 0-1
            targets[zone_id] = target_val

    return groups, targets


def _prepare_quintile_groups(
    df: pd.DataFrame,
    target_col: str,
    quintile_targets_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Prepare 15 zone×quintile supervision groups.

    Uses soft quintile membership (p_q1..p_q5) to assign each cell to its
    most likely quintile group.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling table with 'p_q1'..'p_q5' and 'subregion' columns
    target_col : str
        Target column name (e.g., 'moderate_prevalence')
    quintile_targets_df : pd.DataFrame, optional
        Quintile targets loaded from jam_quintile_targets.csv with columns:
        subregion, quintile, moderate_prevalence, severe_prevalence.
        If None, falls back to zone-level targets for all quintiles.

    Returns
    -------
    groups : np.ndarray, shape [n_cells]
        Integer group IDs (0-14): zone_id * 5 + quintile_id
    targets : dict
        {group_id: target_prevalence} (0-1 scale)
    """
    # Assign each cell to most probable quintile
    quintile_cols = [f'p_q{i}' for i in range(1, 6)]
    most_likely_q = df[quintile_cols].values.argmax(axis=1)  # 0-4

    # Dynamically map zones to integer IDs
    zone_names = sorted(df['subregion'].dropna().unique())
    zone_map = {name: idx for idx, name in enumerate(zone_names)}
    zone_ids = df['subregion'].map(zone_map).values

    # Create 15-group ID: zone * 5 + quintile
    groups = zone_ids * 5 + most_likely_q

    # Build lookup from quintile targets CSV
    qt_lookup = {}  # (subregion, quintile_label) → target value
    if quintile_targets_df is not None and target_col in quintile_targets_df.columns:
        for _, row in quintile_targets_df.iterrows():
            qt_lookup[(row['subregion'], row['quintile'])] = row[target_col]

    # Extract quintile-specific targets
    targets = {}
    quintile_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    for zone_name, zone_id in zone_map.items():
        zone_mask = df['subregion'] == zone_name
        for q in range(1, 6):
            group_id = zone_id * 5 + (q - 1)
            q_label = quintile_labels[q - 1]

            # Look up from quintile targets CSV
            qt_val = qt_lookup.get((zone_name, q_label))
            if qt_val is not None:
                targets[group_id] = qt_val / 100  # Convert % to 0-1
            else:
                # Fallback to zone-level target if quintile not available
                if zone_mask.sum() > 0:
                    target_val = df.loc[zone_mask, target_col].iloc[0] / 100
                    targets[group_id] = target_val

    return groups, targets


def _wsnn_permutation_importance(
    model,
    X: np.ndarray,
    df_mask: pd.DataFrame,
    feature_cols: list[str],
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance for a trained WSNN model.

    Shuffles each feature column n_repeats times and measures the mean increase
    in zone-level MAE vs the baseline (unshuffled) prediction.  A larger MAE
    increase indicates greater importance.

    Parameters
    ----------
    model : WeaklySupervisedNN
        Trained WSNN model for moderate prevalence.
    X : np.ndarray, shape [n_cells, n_features]
        Imputed, unscaled feature matrix (model handles internal scaling).
    df_mask : pd.DataFrame
        Modeling rows (same rows as X).
    feature_cols : list[str]
        Feature names in the same order as X columns.
    n_repeats : int
        Number of shuffle repetitions per feature.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Columns: feature, mean_importance, std_importance, n_repeats
        Sorted descending by mean_importance.
    """
    rng = np.random.default_rng(random_state)

    def _zone_mae(preds: np.ndarray) -> float:
        tmp = df_mask[["subregion", "moderate_prevalence", "population"]].copy()
        tmp["pred"] = preds
        err = 0.0
        n = 0
        for _, grp in tmp.groupby("subregion"):
            w = grp["population"].fillna(0).clip(lower=0).values
            if w.sum() == 0:
                continue
            pred_zone = float(np.average(grp["pred"].values, weights=w))
            true_zone = float(grp["moderate_prevalence"].iloc[0])
            err += abs(pred_zone - true_zone)
            n += 1
        return err / max(n, 1)

    baseline_preds = model.predict(X)
    baseline_mae = _zone_mae(baseline_preds)
    logger.info("WSNN permutation importance — baseline zone MAE: %.3f", baseline_mae)

    rows = []
    for j, feat in enumerate(feature_cols):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_preds = model.predict(X_perm)
            perm_mae = _zone_mae(perm_preds)
            deltas.append(perm_mae - baseline_mae)
        rows.append({
            "feature": feat,
            "mean_importance": float(np.mean(deltas)),
            "std_importance": float(np.std(deltas)),
            "n_repeats": n_repeats,
        })

    fi_df = pd.DataFrame(rows).sort_values("mean_importance", ascending=False).reset_index(drop=True)
    return fi_df


def run(cfg: dict, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Run weakly supervised neural network with aggregation-based training.
    
    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    df : pd.DataFrame
        Modeling table
    
    Returns
    -------
    df : pd.DataFrame
        Updated DataFrame with wsnn_moderate, wsnn_severe, etc. columns
    model_info : dict
        Trained models and metadata
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not installed. Skipping weakly supervised NN.")
        logger.info("Install with: pip install torch")
        return df, {}

    # Avoid thread deadlock with LightGBM/OpenMP when running after GBM
    torch.set_num_threads(1)

    from src.reconciliation.admin_reconcile import reconcile_predictions

    logger.info("Weakly Supervised Neural Network (WSNN) - Aggregation-Based Training")
    
    # Extract configuration
    from src.utils.config_loader import get_available_features
    wsnn_config = cfg.get('modeling', {}).get('weakly_supervised', {})
    feature_cols = get_available_features(cfg, df)
    use_quintile = wsnn_config.get('use_quintile_groups', True)
    
    # Prepare data
    mask = df['in_modeling_sample'].fillna(False)
    X = df.loc[mask, feature_cols].values
    population = df.loc[mask, 'population'].values

    # Impute NaN features (e.g. ~79-86 cells with missing travel times)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    logger.info(f"Training on {mask.sum():,} cells with {len(feature_cols)} features")

    # Load quintile targets CSV if available
    import os
    quintile_targets_df = None
    if use_quintile:
        output_prefix = cfg.get('country', {}).get('output_prefix', 'jam')
        interim_dir = cfg['paths']['interim_dir']
        qt_path = os.path.join(interim_dir, f'{output_prefix}_quintile_targets.csv')
        if os.path.exists(qt_path):
            quintile_targets_df = pd.read_csv(qt_path)
            logger.info(f"Loaded quintile targets from {qt_path} ({len(quintile_targets_df)} rows)")
        else:
            logger.warning(f"Quintile targets not found at {qt_path}, falling back to zone-level targets")

    # Train for both moderate and severe
    models = {}

    for target_type in ['moderate', 'severe']:
        target_col = f'{target_type}_prevalence'
        logger.info(f"\n--- Training for {target_type} prevalence ---")

        # Prepare supervision groups
        if use_quintile:
            logger.info("Using 15 zone×quintile supervision groups")
            supervision_groups, group_targets = _prepare_quintile_groups(
                df[mask], target_col, quintile_targets_df=quintile_targets_df
            )
            logger.info(f"Quintile groups: {len(group_targets)} targets extracted")
        else:
            logger.info("Using 3 zone-level supervision groups")
            supervision_groups, group_targets = _prepare_zone_groups(
                df[mask], target_col
            )
        
        # Initialize model
        model = WeaklySupervisedNN(
            input_dim=X.shape[1],
            hidden_dims=wsnn_config.get('hidden_dims', [64, 32]),
            learning_rate=wsnn_config.get('learning_rate', 0.001),
            dropout=wsnn_config.get('dropout', 0.2)
        )

        # Train with aggregation loss
        model.fit(
            X, population, supervision_groups, group_targets,
            n_epochs=wsnn_config.get('n_epochs', 100),
            verbose=True
        )
        
        # Predict
        raw_preds = model.predict(X)
        df.loc[mask, f'wsnn_raw_{target_type}'] = raw_preds
        
        # Reconcile to exact zone totals
        logger.info(f"Reconciling WSNN {target_type} predictions to official zone targets...")
        df = reconcile_predictions(
            df,
            raw_score_col=f'wsnn_raw_{target_type}',
            target_col=target_col,
            zone_col='subregion',
            population_col='population',
            output_col=f'wsnn_{target_type}',
            strategy='population_weighted',
        )
        
        models[target_type] = model
        logger.info(f"WSNN {target_type} predictions complete.\n")
    
    # Depth metrics (use same approach)
    for target_type in ['moderate', 'severe']:
        depth_col = f'{target_type}_depth'
        logger.info(f"\n--- Training for {target_type} depth ---")
        
        # Prepare groups
        if use_quintile:
            supervision_groups, group_targets = _prepare_quintile_groups(
                df[mask], depth_col, quintile_targets_df=quintile_targets_df
            )
        else:
            supervision_groups, group_targets = _prepare_zone_groups(
                df[mask], depth_col
            )
        
        # Train depth model
        model = WeaklySupervisedNN(
            input_dim=X.shape[1],
            hidden_dims=wsnn_config.get('hidden_dims', [64, 32]),
            learning_rate=wsnn_config.get('learning_rate', 0.001),
            dropout=wsnn_config.get('dropout', 0.2)
        )
        
        model.fit(
            X, population, supervision_groups, group_targets,
            n_epochs=wsnn_config.get('n_epochs', 100),
            verbose=False  # Less verbose for depth
        )
        
        # Predict and reconcile
        raw_preds = model.predict(X)
        df.loc[mask, f'wsnn_raw_{target_type}_depth'] = raw_preds
        
        df = reconcile_predictions(
            df,
            raw_score_col=f'wsnn_raw_{target_type}_depth',
            target_col=depth_col,
            zone_col='subregion',
            population_col='population',
            output_col=f'wsnn_{target_type}_depth',
            strategy='population_weighted',
        )
        
        models[f'{target_type}_depth'] = model
    
    logger.info("Weakly supervised NN complete.")

    # --- Permutation importance (E5) ---
    # Uses the moderate-prevalence model to compute feature-drop MAE scores.
    try:
        n_repeats = cfg.get("modeling", {}).get("weakly_supervised", {}).get(
            "n_importance_repeats", 30
        )
        fi_df = _wsnn_permutation_importance(
            models["moderate"], X, df[mask], feature_cols,
            n_repeats=n_repeats, random_state=42
        )
        eval_dir = cfg["paths"]["eval_dir"]
        os.makedirs(eval_dir, exist_ok=True)
        prefix = cfg.get("country", {}).get("output_prefix", "nga")
        fi_path = os.path.join(eval_dir, f"{prefix}_wsnn_importance.csv")
        fi_df.to_csv(fi_path, index=False)
        logger.info(
            "WSNN permutation importance saved: %s\n%s",
            fi_path,
            fi_df.head(10).to_string(index=False),
        )
    except Exception as e:
        logger.warning("WSNN permutation importance failed (non-fatal): %s", e)
        fi_df = None

    return df, {
        'models': models,
        'feature_cols': feature_cols,
        'supervision_type': '15 quintile groups' if use_quintile else '3 zone groups',
        'permutation_importance': fi_df,
    }
