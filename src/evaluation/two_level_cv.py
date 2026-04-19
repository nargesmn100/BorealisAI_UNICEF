"""
Two-Level Cross-Validation Framework
======================================

Per problem statement §9, evaluates geographic generalization at two levels:

Level 1 — Hold out full states/zones (extends existing LOZO).
Level 2 — Within training states, hold out urban vs rural cells.
           Train on one setting (e.g., rural-only) in training states,
           predict on: (a) all cells in held-out state, (b) the held-out
           setting (e.g., urban) in training states.

This tests both inter-region and intra-region generalization, which is
critical for Nigeria where urban/rural deprivation patterns differ strongly.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils.config_loader import get_available_features

logger = logging.getLogger(__name__)

# Import from shared location — do not duplicate here
from src.utils.admin_mappings import NIGERIA_GEOPOLITICAL_ZONES


def two_level_cross_validation(
    df: pd.DataFrame,
    cfg: dict,
    max_folds: int = 10,
) -> pd.DataFrame:
    """
    Two-level cross-validation: geographic + urban/rural holdout.

    Level 1: Hold out one state at a time (LOZO).
    Level 2: Within training states, hold out urban OR rural cells.
             Evaluate on (a) held-out state, (b) held-out setting in training states.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling table with features, targets, is_urban, and subregion columns.
    cfg : dict
        Loaded config.
    max_folds : int
        Maximum number of state holdout folds (performance guard).

    Returns
    -------
    pd.DataFrame
        Results with columns: held_out_state, held_out_setting, test_set,
        method, n_test, predicted_aggregate, target, abs_error.
    """
    feature_cols = get_available_features(cfg, df)
    zone_col = cfg["modeling"]["admin_zone_col"]
    pop_col = "population"
    target_col = "moderate_prevalence"
    ridge_cfg = cfg["modeling"]["ridge"]

    model_mask = df["in_modeling_sample"].fillna(False)
    # Impute features upfront
    feature_mask = df[feature_cols].notna().all(axis=1)
    valid_mask = model_mask & feature_mask

    if "is_urban" not in df.columns:
        logger.warning("is_urban column not found. Skipping two-level CV.")
        return pd.DataFrame()

    states = sorted(df.loc[valid_mask, zone_col].unique())
    if len(states) < 3:
        logger.warning(
            "Two-level CV requires at least 3 states, found %d. Skipping.",
            len(states),
        )
        return pd.DataFrame()

    # Subsample states if needed
    if len(states) > max_folds:
        rng = np.random.RandomState(42)
        states = sorted(rng.choice(states, size=max_folds, replace=False))
        logger.info("Subsampled to %d states for two-level CV.", len(states))

    logger.info(
        "Running two-level CV: %d states × 2 urban/rural settings", len(states)
    )

    rows = []

    for held_out_state in states:
        state_test_mask = valid_mask & (df[zone_col] == held_out_state)
        state_train_mask = valid_mask & (df[zone_col] != held_out_state)

        if state_test_mask.sum() < 5 or state_train_mask.sum() < 10:
            continue

        target_value = df.loc[state_test_mask, target_col].iloc[0]

        # Level 2: hold out urban (is_urban==1) or rural (is_urban==0) within training
        for urban_val in [0, 1]:
            setting_name = "urban" if urban_val == 1 else "rural"

            # Train on training states, EXCLUDING the held-out setting
            train_mask = state_train_mask & (df["is_urban"] != urban_val)
            # Test sets:
            # (a) All cells in held-out state
            test_state_mask = state_test_mask
            # (b) Held-out setting cells in training states
            test_within_mask = state_train_mask & (df["is_urban"] == urban_val)

            if train_mask.sum() < 5:
                continue

            X_train = df.loc[train_mask, feature_cols].values.astype(float)
            y_train = df.loc[train_mask, target_col].values.astype(float)

            # Fit Ridge
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(
                    alphas=ridge_cfg["alpha_candidates"],
                    cv=min(ridge_cfg["cv_folds"], max(2, len(np.unique(y_train)))),
                    scoring="neg_mean_squared_error",
                )),
            ])
            pipe.fit(X_train, y_train)

            # Evaluate on held-out state
            if test_state_mask.sum() > 0:
                X_test = df.loc[test_state_mask, feature_cols].values.astype(float)
                preds = pipe.predict(X_test)
                pop = df.loc[test_state_mask, pop_col].values.astype(float)
                pop = np.where(np.isnan(pop) | (pop < 0), 0.0, pop)
                agg = np.average(preds, weights=pop) if pop.sum() > 0 else preds.mean()
                rows.append({
                    "held_out_state": held_out_state,
                    "held_out_setting": setting_name,
                    "test_set": "held_out_state",
                    "method": "ridge",
                    "n_test": int(test_state_mask.sum()),
                    "predicted_aggregate": round(float(agg), 2),
                    "target": round(float(target_value), 2),
                    "abs_error": round(abs(float(agg) - float(target_value)), 2),
                })

            # Evaluate on held-out setting within training states
            if test_within_mask.sum() > 0:
                X_within = df.loc[test_within_mask, feature_cols].values.astype(float)
                preds_w = pipe.predict(X_within)
                pop_w = df.loc[test_within_mask, pop_col].values.astype(float)
                pop_w = np.where(np.isnan(pop_w) | (pop_w < 0), 0.0, pop_w)
                agg_w = np.average(preds_w, weights=pop_w) if pop_w.sum() > 0 else preds_w.mean()
                # For within-training test, compute the actual weighted target
                within_targets = df.loc[test_within_mask, target_col].values.astype(float)
                actual_w = np.average(within_targets, weights=pop_w) if pop_w.sum() > 0 else within_targets.mean()
                rows.append({
                    "held_out_state": held_out_state,
                    "held_out_setting": setting_name,
                    "test_set": f"within_training_{setting_name}",
                    "method": "ridge",
                    "n_test": int(test_within_mask.sum()),
                    "predicted_aggregate": round(float(agg_w), 2),
                    "target": round(float(actual_w), 2),
                    "abs_error": round(abs(float(agg_w) - float(actual_w)), 2),
                })

    result = pd.DataFrame(rows)

    if len(result) > 0:
        # Summary statistics
        for test_set in result["test_set"].unique():
            subset = result[result["test_set"] == test_set]
            logger.info(
                "Two-level CV [%s]: mean abs error = %.2f pp (n=%d folds)",
                test_set, subset["abs_error"].mean(), len(subset),
            )
    else:
        logger.warning("Two-level CV produced no results.")

    return result


def add_geopolitical_zones(df: pd.DataFrame, zone_col: str = "subregion") -> pd.DataFrame:
    """
    Add Nigeria geopolitical zone column based on state name.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a column with state names (title case).
    zone_col : str
        Column containing state names.

    Returns
    -------
    pd.DataFrame
        With added 'geopolitical_zone' column.
    """
    df = df.copy()
    df["geopolitical_zone"] = df[zone_col].map(NIGERIA_GEOPOLITICAL_ZONES).fillna("Unknown")
    n_mapped = (df["geopolitical_zone"] != "Unknown").sum()
    logger.info(
        "Geopolitical zone mapping: %d/%d cells mapped to zones.",
        n_mapped, len(df),
    )
    return df
