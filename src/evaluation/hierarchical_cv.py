"""
Hierarchical Cross-Level Validation
=====================================

Core scientific test: train a model at a coarse geographic level (e.g. 6
geopolitical zones), predict all grid cells, aggregate predictions to a finer
level (e.g. 37 states), and compare against independently computed MICS
targets at that finer level.

If the model trained on 6 zones recovers 37 state patterns it never saw
during training → the methodology genuinely learns spatial variation from
features, not just post-hoc rescaling.

Three experiments (configurable via config_nga.yaml):
    A. train: geopolitical_zone (6)  →  eval: state (37)
    B. train: geopolitical_zone (6)  →  eval: state_urban_rural (74)
    C. train: state (37)             →  eval: state_urban_rural (74)

For each experiment × model:
    1. Assign each cell the training-level target as its label.
    2. Train model on features → training-level labels.
    3. Predict all cells (raw, no reconciliation).
    4. Optionally reconcile to training-level zone totals.
    5. Aggregate predictions (population-weighted) to eval-level groups.
    6. Compare aggregated predictions to held-out MICS eval-level targets.
    7. Report MAE, RMSE, Pearson r, Spearman r.

Outputs
-------
    data/outputs/nga/eval/hierarchical_validation.csv        — summary per experiment
    data/outputs/nga/eval/hierarchical_validation_detail.csv — per-group breakdown
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.reconciliation.admin_reconcile import reconcile_predictions
from src.utils.config_loader import get_available_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pop_weighted_aggregate(
    predictions: np.ndarray,
    populations: np.ndarray,
) -> float:
    """Population-weighted mean of predictions."""
    pop = np.where(np.isnan(populations) | (populations < 0), 0.0, populations)
    if pop.sum() == 0:
        return float(predictions.mean())
    return float(np.average(predictions, weights=pop))


def _metrics(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """Compute MAE, RMSE, Pearson r, Spearman r between two arrays."""
    n = len(predicted)
    mae = float(np.mean(np.abs(predicted - actual)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))

    if n >= 3:
        pearson_r, _ = pearsonr(predicted, actual)
        spearman_r, _ = spearmanr(predicted, actual)
    else:
        pearson_r = float("nan")
        spearman_r = float("nan")

    return {
        "n_groups": n,
        "mae_pp": round(mae, 3),
        "rmse_pp": round(rmse, 3),
        "pearson_r": round(pearson_r, 3) if not np.isnan(pearson_r) else None,
        "spearman_r": round(spearman_r, 3) if not np.isnan(spearman_r) else None,
    }


def _train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_all: np.ndarray,
    alpha_candidates: list,
    cv_folds: int,
) -> np.ndarray:
    """Fit a Ridge pipeline and predict on X_all."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(
            alphas=alpha_candidates,
            cv=min(cv_folds, max(2, len(np.unique(y_train)))),
            scoring="neg_mean_squared_error",
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_all)


def _train_gbm(cfg: dict, X_train: np.ndarray, y_train: np.ndarray,
               X_all: np.ndarray) -> np.ndarray | None:
    """Fit LightGBM and predict on X_all. Returns None if unavailable.

    Forces n_jobs=1 to prevent OpenMP thread contention when called
    inside a loop (hierarchical CV experiments).
    """
    try:
        from src.models.gbm_model import _get_gbm_model
        import os
        # Suppress OpenMP thread conflicts that occur in nested loop contexts
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("LIGHTGBM_NUM_THREADS", "1")

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_train)
        X_al = imp.transform(X_all)

        # Override n_jobs to 1 for thread safety in this context
        cfg_copy = cfg.copy()
        cfg_copy["modeling"] = {**cfg["modeling"]}
        cfg_copy["modeling"]["gbm"] = {**cfg["modeling"]["gbm"], "n_jobs": 1}

        gbm = _get_gbm_model(cfg_copy)
        gbm.fit(X_tr, y_train)
        return gbm.predict(X_al)
    except Exception as e:
        logger.warning("GBM unavailable for hierarchical CV: %s", e)
        return None


# ---------------------------------------------------------------------------
# Level-column resolver
# ---------------------------------------------------------------------------

def _level_col(level: str) -> str:
    """Map level name to the modeling-table column that carries that label."""
    mapping = {
        "geopolitical_zone": "geopolitical_zone",
        "state": "subregion",
        "state_urban_rural": "state_urban_rural",
    }
    if level not in mapping:
        raise ValueError(
            f"Unknown hierarchical level '{level}'. "
            f"Expected one of: {list(mapping)}"
        )
    return mapping[level]


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def _run_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_level: str,
    eval_level: str,
    eval_targets: pd.DataFrame,
    cfg: dict,
    models: list[str],
    reconcile: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Run one hierarchical experiment.

    Parameters
    ----------
    df : pd.DataFrame
        Full modeling table (all cells).
    feature_cols : list[str]
        Feature columns to use.
    train_level : str
        Level name to train on ("geopolitical_zone", "state", etc.)
    eval_level : str
        Level name to evaluate at.
    eval_targets : pd.DataFrame
        Ground-truth targets at eval level. Must have columns:
        group_id, moderate_prevalence.
    cfg : dict
        Config.
    models : list[str]
        Which models to run, e.g. ["ridge", "gbm"].
    reconcile : bool
        Whether to also report reconciled predictions.

    Returns
    -------
    summary_rows : list[dict]
        One dict per (model, reconciled) pair.
    detail_rows : list[dict]
        One dict per (model, reconciled, eval_group).
    """
    train_col = _level_col(train_level)
    eval_col = _level_col(eval_level)
    pop_col = "population"

    mask = df["in_modeling_sample"].fillna(False)
    valid = mask & df[feature_cols].notna().all(axis=1)

    # Use a reset-index slice so all positional arrays stay aligned
    valid_df = df.loc[valid].reset_index(drop=True)
    X_all = valid_df[feature_cols].values.astype(float)
    pop_all = valid_df[pop_col].values.astype(float)

    # Training labels = training-level moderate_prevalence per cell
    if train_col not in valid_df.columns:
        logger.warning("Column '%s' not in modeling table. Skipping experiment.", train_col)
        return [], []

    # Build per-cell training label: each cell gets its training-level group target
    train_labels = valid_df["moderate_prevalence"].values.astype(float)

    if train_level != "state":
        # Map each cell to its coarser training-level group target
        grp_targets = (
            valid_df.groupby(train_col)["moderate_prevalence"]
            .first()
            .to_dict()
        )
        train_labels = valid_df[train_col].map(grp_targets).values.astype(float)
    else:
        grp_targets = {}  # not needed for state-level training

    # Filter out rows with NaN training labels
    label_valid = ~np.isnan(train_labels)
    X_train = X_all[label_valid]
    y_train = train_labels[label_valid]

    if len(X_train) < 10:
        logger.warning("Insufficient training data for experiment %s→%s. Skipping.",
                       train_level, eval_level)
        return [], []

    logger.info("Experiment %s → %s | train: %d cells, features: %d",
                train_level, eval_level, len(X_train), len(feature_cols))

    # Eval targets lookup
    eval_lookup = dict(zip(eval_targets["group_id"], eval_targets["moderate_prevalence"]))

    # Eval-level column for grouping predictions
    if eval_col not in valid_df.columns:
        logger.warning("Eval column '%s' not in modeling table. Skipping.", eval_col)
        return [], []

    eval_groups_all = valid_df[eval_col].values

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    ridge_cfg = cfg["modeling"]["ridge"]

    for model_name in models:
        # --- Train ---
        if model_name == "ridge":
            raw_preds = _train_ridge(
                X_train, y_train, X_all,
                alpha_candidates=ridge_cfg["alpha_candidates"],
                cv_folds=ridge_cfg["cv_folds"],
            )
        elif model_name == "gbm":
            raw_preds = _train_gbm(cfg, X_train, y_train, X_all)
            if raw_preds is None:
                continue
        else:
            logger.warning("Unknown model '%s'. Skipping.", model_name)
            continue

        for do_reconcile in ([False, True] if reconcile else [False]):
            label = "reconciled" if do_reconcile else "raw"

            preds = raw_preds.copy()

            if do_reconcile:
                # Reconcile to training-level zone totals.
                # Work entirely in the DataFrame to avoid pandas-label vs
                # numpy-position index confusion.
                tmp = valid_df.copy()
                tmp["_raw"] = preds
                tmp["_pop"] = pop_all
                for zone, grp_idx in tmp.groupby(train_col).groups.items():
                    zone_target = grp_targets.get(zone) if train_level != "state" else None
                    if zone_target is None:
                        # For state-level training, pull from the modeling table
                        state_rows = tmp.loc[grp_idx, "moderate_prevalence"]
                        zone_target = state_rows.iloc[0] if len(state_rows) > 0 else None
                    if zone_target is None:
                        continue
                    pop_z = tmp.loc[grp_idx, "_pop"].values.astype(float)
                    pop_z = np.where(np.isnan(pop_z) | (pop_z < 0), 0, pop_z)
                    pred_z = tmp.loc[grp_idx, "_raw"].values.astype(float)
                    if pop_z.sum() > 0:
                        achieved = np.average(pred_z, weights=pop_z)
                    else:
                        achieved = pred_z.mean()
                    if achieved > 0:
                        tmp.loc[grp_idx, "_raw"] = pred_z * (zone_target / achieved)
                preds = tmp["_raw"].values

            # --- Aggregate predictions to eval level ---
            pred_agg: list[float] = []
            true_agg: list[float] = []
            group_ids: list[str] = []

            for eg, eg_idx in pd.Series(eval_groups_all).groupby(eval_groups_all).groups.items():
                if str(eg) not in eval_lookup:
                    continue
                eg_idx = eg_idx.values
                eg_pop = pop_all[eg_idx]
                eg_pred = preds[eg_idx]
                pred_mean = _pop_weighted_aggregate(eg_pred, eg_pop)
                true_val = eval_lookup[str(eg)]
                pred_agg.append(pred_mean)
                true_agg.append(true_val)
                group_ids.append(str(eg))

                detail_rows.append({
                    "train_level": train_level,
                    "eval_level": eval_level,
                    "model": model_name,
                    "reconciled": do_reconcile,
                    "group_id": str(eg),
                    "predicted": round(pred_mean, 3),
                    "target": round(true_val, 3),
                    "error": round(pred_mean - true_val, 3),
                    "abs_error": round(abs(pred_mean - true_val), 3),
                    "n_cells": len(eg_idx),
                })

            if len(pred_agg) < 2:
                logger.warning("Not enough eval groups for metrics (%d). Skipping.", len(pred_agg))
                continue

            m = _metrics(np.array(pred_agg), np.array(true_agg))
            summary_rows.append({
                "train_level": train_level,
                "eval_level": eval_level,
                "model": model_name,
                "reconciled": do_reconcile,
                **m,
            })

            logger.info(
                "  [%s → %s | %s | %s] MAE=%.2f pp | RMSE=%.2f pp | "
                "Pearson=%.3f | Spearman=%.3f | n_groups=%d",
                train_level, eval_level, model_name, label,
                m["mae_pp"], m["rmse_pp"],
                m["pearson_r"] or float("nan"),
                m["spearman_r"] or float("nan"),
                m["n_groups"],
            )

    return summary_rows, detail_rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hierarchical_validation(
    df: pd.DataFrame,
    cfg: dict,
    interim_dir: str,
    eval_dir: str,
) -> pd.DataFrame:
    """
    Run all configured hierarchical validation experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Nigeria modeling table (output of step05, must have geopolitical_zone
        and state_urban_rural columns).
    cfg : dict
        Loaded config (config_nga.yaml). Reads:
            evaluation.hierarchical_cv.experiments
            evaluation.hierarchical_cv.models
    interim_dir : str
        Path to nga/interim — where multilevel target CSVs live.
    eval_dir : str
        Path where output CSVs will be saved.

    Returns
    -------
    pd.DataFrame
        Summary results table.
    """
    hcv_cfg = cfg.get("evaluation", {}).get("hierarchical_cv", {})
    if not hcv_cfg.get("enabled", False):
        logger.info("Hierarchical CV disabled in config. Skipping.")
        return pd.DataFrame()

    experiments = hcv_cfg.get("experiments", [])
    models = hcv_cfg.get("models", ["ridge"])
    feature_cols = get_available_features(cfg, df)

    if not experiments:
        logger.warning("No hierarchical_cv experiments configured. Skipping.")
        return pd.DataFrame()

    logger.info("=" * 60)
    logger.info("Hierarchical Cross-Level Validation")
    logger.info("%d experiments × %d models", len(experiments), len(models))
    logger.info("=" * 60)

    # Load all multilevel target files once
    level_targets: dict[str, pd.DataFrame] = {}
    for level_name in ["national", "geopolitical_zone", "state", "state_urban_rural"]:
        path = os.path.join(interim_dir, f"nga_targets_{level_name}.csv")
        if os.path.isfile(path):
            level_targets[level_name] = pd.read_csv(path)
            logger.info("Loaded %s targets: %d groups", level_name, len(level_targets[level_name]))
        else:
            logger.warning("Target file not found: %s", path)

    all_summary: list[dict] = []
    all_detail: list[dict] = []

    for exp in experiments:
        train_level = exp["train_level"]
        eval_level = exp["eval_level"]

        if eval_level not in level_targets:
            logger.warning("No target file for eval_level='%s'. Skipping.", eval_level)
            continue

        eval_targets = level_targets[eval_level]
        summary_rows, detail_rows = _run_experiment(
            df=df,
            feature_cols=feature_cols,
            train_level=train_level,
            eval_level=eval_level,
            eval_targets=eval_targets,
            cfg=cfg,
            models=models,
            reconcile=True,
        )
        all_summary.extend(summary_rows)
        all_detail.extend(detail_rows)

    os.makedirs(eval_dir, exist_ok=True)
    summary_df = pd.DataFrame(all_summary)
    detail_df = pd.DataFrame(all_detail)

    if not summary_df.empty:
        summary_path = os.path.join(eval_dir, "hierarchical_validation.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info("Hierarchical validation summary saved to: %s", summary_path)

        detail_path = os.path.join(eval_dir, "hierarchical_validation_detail.csv")
        detail_df.to_csv(detail_path, index=False)
        logger.info("Hierarchical validation detail saved to: %s", detail_path)

        logger.info("\n=== Hierarchical Validation Summary ===")
        for _, row in summary_df.iterrows():
            logger.info(
                "  %s → %s | %s | reconciled=%s | MAE=%.2f pp | Pearson=%.3f | Spearman=%.3f",
                row["train_level"], row["eval_level"], row["model"],
                row["reconciled"], row["mae_pp"],
                row["pearson_r"] if row["pearson_r"] is not None else float("nan"),
                row["spearman_r"] if row["spearman_r"] is not None else float("nan"),
            )
    else:
        logger.warning("Hierarchical validation produced no results.")

    return summary_df
