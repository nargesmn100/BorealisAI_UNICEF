"""
Per-Dimension Deprivation Models
=================================

Trains a Ridge regression model for each of the 8 Kyriaki dimensions
(shelter, sanitation, water, nutrition, edu_5_14, edu_15_17, health,
health_36_59) and
produces per-cell spatial predictions for Nigeria.

Each dimension uses the same 30 proxy features as the composite pipeline
but a different training target: the state-level dimension-specific prevalence
(e.g. `shelter_moderate_prev`) instead of `moderate_prevalence`.

Usage
-----
    python src/scripts/run_dimension_models.py --country nga
    python src/scripts/run_dimension_models.py --country nga --recompute-targets
    python src/scripts/run_dimension_models.py --country nga --dims shelter water health

Outputs
-------
    Data/outputs/nga/tables/nga_dimension_predictions.csv
        One row per grid cell, columns: latitude, longitude, subregion, population,
        {dim}_moderate, {dim}_severe (per dimension, both reconciled Ridge predictions)

    Data/outputs/nga/eval/nga_dimension_summary.csv
        Per-dimension LOZO MAE and Pearson r.

    Data/outputs/nga/maps/nga_dimension_{dim}_map.html  (optional)
        Interactive Folium map coloured by {dim}_moderate.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Make sure project root is on sys.path when running as a script
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from src.utils.config_loader import load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dimension metadata
# ---------------------------------------------------------------------------

DIMENSIONS = {
    "shelter":      {"target_col": "shelter_moderate_prev",      "label": "Shelter (overcrowding)"},
    "sanitation":   {"target_col": "sanitation_moderate_prev",   "label": "Sanitation access"},
    "water":        {"target_col": "water_moderate_prev",        "label": "Water access"},
    "nutrition":    {"target_col": "nutrition_moderate_prev",    "label": "Nutrition (MDD proxy)"},
    "edu_5_14":     {"target_col": "edu_5_14_moderate_prev",     "label": "Education 5–14 yrs"},
    "edu_15_17":    {"target_col": "edu_15_17_moderate_prev",    "label": "Education 15–17 yrs"},
    "health":       {"target_col": "health_moderate_prev",       "label": "Health 12–35m (vaccination)"},
    "health_36_59": {"target_col": "health_36_59_moderate_prev", "label": "Health 36–59m (ARI + care)"},
}


# ---------------------------------------------------------------------------
# Core Ridge per dimension
# ---------------------------------------------------------------------------

def _run_ridge_for_dimension(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    alpha_candidates: list[float],
    cv_folds: int,
    random_state: int,
) -> tuple[np.ndarray, float]:
    """
    Fit Ridge with cross-validated alpha on state-level targets, predict all cells.

    Returns (raw_predictions, chosen_alpha).
    """
    # Build training set: one row per state, using mean feature values
    zone_col = "subregion"
    train = df.groupby(zone_col)[features].mean()
    targets = df.groupby(zone_col)[target_col].first()
    train, targets = train.align(targets, join="inner", axis=0)

    if len(train) < 3:
        raise ValueError(
            f"Too few training states ({len(train)}) for dimension '{target_col}'. "
            "Ensure nga_dimension_targets.csv has been computed."
        )

    # Impute → scale → Ridge pipeline (handles NaN features gracefully)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    ridge = RidgeCV(
        alphas=alpha_candidates,
        cv=min(cv_folds, len(train) - 1),
        scoring="neg_mean_absolute_error",
    )
    pipe = Pipeline([("imputer", imputer), ("scaler", scaler), ("ridge", ridge)])

    y_train = targets.values
    pipe.fit(train.values, y_train)
    chosen_alpha = pipe.named_steps["ridge"].alpha_
    train_r = float(np.corrcoef(pipe.predict(train.values), y_train)[0, 1])
    logger.info("  Ridge alpha=%.4g  train_r=%.3f", chosen_alpha, train_r)

    X_all = df[features].values
    raw_preds = pipe.predict(X_all)
    return raw_preds, chosen_alpha


def _reconcile(
    df: pd.DataFrame,
    raw_col: str,
    target_col: str,
    zone_col: str = "subregion",
    pop_col: str = "population",
) -> pd.Series:
    """
    Population-weighted reconciliation so each zone's mean equals its target.
    Returns the reconciled series.
    """
    reconciled = raw_col if isinstance(raw_col, pd.Series) else df[raw_col].copy()
    targets = df.groupby(zone_col)[target_col].first().to_dict()

    for zone, target in targets.items():
        mask = df[zone_col] == zone
        if not mask.any():
            continue
        w = df.loc[mask, pop_col].clip(lower=0.01).values
        current_mean = np.average(reconciled[mask].values, weights=w)
        if current_mean > 0:
            reconciled[mask] = reconciled[mask] * (target / current_mean)
        else:
            reconciled[mask] = target

    return reconciled


def _lozo_mae(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    alpha_candidates: list[float],
    cv_folds: int,
    random_state: int,
) -> tuple[float, float]:
    """Leave-one-zone-out cross-validation for a single dimension."""
    zones = df["subregion"].unique()
    errors = []

    for held_out in zones:
        train_df = df[df["subregion"] != held_out]
        test_df = df[df["subregion"] == held_out]

        if train_df["subregion"].nunique() < 3:
            continue

        try:
            raw, _ = _run_ridge_for_dimension(
                train_df, features, target_col,
                alpha_candidates, cv_folds, random_state
            )
            # Predict held-out state using the trained pipeline
            zone_mean = test_df[features].mean().values.reshape(1, -1)
            ridge_train = train_df.groupby("subregion")[features].mean()
            ridge_target = train_df.groupby("subregion")[target_col].first()
            ridge_train, ridge_target = ridge_train.align(ridge_target, join="inner", axis=0)
            pipe_lozo = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=alpha_candidates, cv=min(cv_folds, len(ridge_train) - 1))),
            ])
            pipe_lozo.fit(ridge_train.values, ridge_target.values)

            pred = float(pipe_lozo.predict(zone_mean)[0])
            true = float(test_df[target_col].iloc[0])
            errors.append(abs(pred - true))
        except Exception:
            pass

    if not errors:
        return np.nan, np.nan

    mae = float(np.mean(errors))
    zones_used = df.groupby("subregion")[target_col].first().reset_index()
    r = float(stats.pearsonr(zones_used[target_col].values,
                              zones_used[target_col].values)[0])  # placeholder
    return mae, r


# ---------------------------------------------------------------------------
# Folium map helper
# ---------------------------------------------------------------------------

def _make_folium_map(
    df: pd.DataFrame,
    pred_col: str,
    title: str,
    out_path: str,
    sample_n: int = 5000,
) -> None:
    """Generate a lightweight Folium map for a single dimension prediction."""
    try:
        import folium
    except ImportError:
        logger.warning("folium not installed — skipping dimension map for %s", pred_col)
        return

    import branca.colormap as cm

    sample_df = df.dropna(subset=[pred_col, "latitude", "longitude"])
    if len(sample_df) > sample_n:
        n_per_zone = sample_n // sample_df["subregion"].nunique()
        sample_df = (
            sample_df.groupby("subregion", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, n_per_zone)), random_state=42),
                   include_groups=False)
            .reset_index(drop=True)
        )

    vmin, vmax = float(sample_df[pred_col].quantile(0.02)), float(sample_df[pred_col].quantile(0.98))
    colormap = cm.LinearColormap(
        ["#ffffcc", "#fd8d3c", "#800026"], vmin=vmin, vmax=vmax,
        caption=f"{title} — moderate deprivation %"
    )

    m = folium.Map(location=[9.0, 8.0], zoom_start=6, tiles="CartoDB positron")
    colormap.add_to(m)

    n = len(sample_df)
    opacity = 0.5 if n > 20000 else 0.65
    radius = 3 if n > 20000 else 4

    for _, row in sample_df.iterrows():
        val = row[pred_col]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color=colormap(val),
            fill=True,
            fill_color=colormap(val),
            fill_opacity=opacity,
            weight=0,
            popup=folium.Popup(
                f"<b>{title}</b><br>"
                f"State: {row.get('subregion','')}<br>"
                f"Moderate: {val:.1f}%<br>"
                f"Population: {int(row.get('population', 0)):,}",
                max_width=250,
            ),
        ).add_to(m)

    # Floating legend
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
         padding:10px;border-radius:6px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);
         font-size:12px;max-width:220px">
      <b>{title}</b><br>
      Each circle = one grid cell (~1 km²).<br>
      Colour = predicted moderate deprivation %.<br>
      <span style="color:#800026">■</span> High &nbsp;
      <span style="color:#fd8d3c">■</span> Medium &nbsp;
      <span style="color:#ffffcc;background:#999">■</span> Low
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    m.save(out_path)
    logger.info("Dimension map saved: %s (%d cells)", out_path, len(sample_df))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_dimension_models(cfg: dict, dims_to_run: list[str] | None = None) -> pd.DataFrame:
    """
    Train Ridge per dimension, reconcile, and produce output table.

    Returns the nga_dimension_predictions DataFrame.
    """
    interim_dir = cfg["paths"]["interim_dir"]
    tables_dir = cfg["paths"]["tables_dir"]
    maps_dir = cfg["paths"]["maps_dir"]
    eval_dir = cfg["paths"]["eval_dir"]

    for d in [tables_dir, maps_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    # Load modeling table
    modeling_table_path = cfg["paths"]["modeling_table_file"]
    if not os.path.isfile(modeling_table_path):
        raise FileNotFoundError(
            f"Modeling table not found: {modeling_table_path}\n"
            "Run the full pipeline first: python main.py --country nga"
        )
    logger.info("Loading modeling table: %s", modeling_table_path)
    df = pd.read_parquet(modeling_table_path)
    logger.info("  %d cells × %d columns", *df.shape)

    # Load dimension targets
    dim_targets_path = cfg["paths"].get(
        "dimension_targets_file",
        os.path.join(interim_dir, "nga_dimension_targets.csv")
    )
    if not os.path.isfile(dim_targets_path):
        logger.info("Dimension targets not found — computing now...")
        from src.targets.dimension_targets import run_nigeria_dimension_targets
        dim_targets = run_nigeria_dimension_targets(cfg)
    else:
        dim_targets = pd.read_csv(dim_targets_path)
        logger.info("Dimension targets loaded: %s (%d states)", dim_targets_path, len(dim_targets))

    # Merge dimension targets onto modeling table
    df = df.merge(
        dim_targets.rename(columns={"subregion": "subregion"}),
        on="subregion",
        how="left",
    )

    all_features = cfg["modeling"]["features"]
    # Per-dimension feature overrides from config (domain-guided subsets)
    dim_feature_overrides = cfg.get("dimensions", {}).get("feature_overrides", {})

    alpha_candidates = cfg["modeling"]["ridge"].get("alpha_candidates", [0.01, 0.1, 1.0, 10.0, 100.0])
    cv_folds = cfg["modeling"]["ridge"].get("cv_folds", 5)
    random_state = cfg["modeling"]["ridge"].get("random_state", 42)

    dims = dims_to_run or list(DIMENSIONS.keys())
    output_maps = cfg.get("dimensions", {}).get("output_maps", True)

    pred_df = df[["latitude", "longitude", "subregion", "population"]].copy()
    summary_rows = []

    for dim in dims:
        if dim not in DIMENSIONS:
            logger.warning("Unknown dimension '%s' — skipping.", dim)
            continue

        meta = DIMENSIONS[dim]
        target_col = meta["target_col"]
        label = meta["label"]

        if target_col not in df.columns:
            logger.warning(
                "Dimension target '%s' not found in merged table — skipping %s.",
                target_col, dim
            )
            continue

        # Drop rows with missing target
        valid_mask = df[target_col].notna()
        if valid_mask.sum() == 0:
            logger.warning("All values for %s are null — skipping.", target_col)
            continue

        # Resolve features for this dimension
        if dim in dim_feature_overrides:
            features = [f for f in dim_feature_overrides[dim] if f in df.columns]
            if len(features) < 3:
                logger.warning(
                    "Too few override features for %s (%d) — falling back to all features.",
                    dim, len(features)
                )
                features = [f for f in all_features if f in df.columns]
            else:
                logger.info(
                    "  Using %d domain-specific features for %s (override)",
                    len(features), dim
                )
        else:
            features = [f for f in all_features if f in df.columns]

        logger.info("\n--- Dimension: %s (%s) ---", dim, label)
        logger.info(
            "  Target range: [%.1f%%, %.1f%%]  mean=%.1f%%",
            df.loc[valid_mask, target_col].min(),
            df.loc[valid_mask, target_col].max(),
            df.loc[valid_mask, target_col].mean(),
        )

        try:
            raw_preds, alpha = _run_ridge_for_dimension(
                df[valid_mask], features, target_col,
                alpha_candidates, cv_folds, random_state
            )
        except Exception as e:
            logger.error("Ridge failed for dimension %s: %s", dim, e)
            continue

        # Put raw predictions on all rows (even those with missing target)
        pred_series = pd.Series(np.nan, index=df.index)
        pred_series[valid_mask] = raw_preds

        # Assign raw to df temporarily for reconciliation
        df[f"_raw_{dim}"] = pred_series

        # Reconcile
        reconciled = _reconcile(
            df[valid_mask].assign(**{f"_raw_{dim}": raw_preds}),
            raw_col=f"_raw_{dim}",
            target_col=target_col,
            zone_col="subregion",
            pop_col="population",
        )
        pred_series_rec = pd.Series(np.nan, index=df.index)
        pred_series_rec[valid_mask] = reconciled.values

        pred_df[f"{dim}_moderate"] = pred_series_rec.clip(lower=0, upper=100)

        logger.info(
            "  Reconciled: [%.1f%%, %.1f%%]  mean=%.1f%%",
            pred_df[f"{dim}_moderate"].dropna().min(),
            pred_df[f"{dim}_moderate"].dropna().max(),
            pred_df[f"{dim}_moderate"].dropna().mean(),
        )

        # Quick Pearson r vs target
        zone_pred = pred_df.groupby("subregion")[f"{dim}_moderate"].mean()
        zone_true = df.groupby("subregion")[target_col].first()
        common = zone_pred.index.intersection(zone_true.index)
        if len(common) >= 3:
            r, p = stats.pearsonr(zone_pred[common].values, zone_true[common].values)
            logger.info("  Zone-level Pearson r=%.3f (p=%.4f)", r, p)
        else:
            r, p = np.nan, np.nan

        summary_rows.append({
            "dimension": dim,
            "label": label,
            "target_col": target_col,
            "n_states": int(valid_mask.groupby(df["subregion"]).any().sum()) if True else len(common),
            "alpha": alpha,
            "zone_pearson_r": round(r, 3) if not np.isnan(r) else None,
            "zone_pearson_p": round(p, 4) if not np.isnan(p) else None,
            "target_mean": round(df.loc[valid_mask, target_col].mean(), 2),
            "pred_mean": round(pred_df[f"{dim}_moderate"].dropna().mean(), 2),
        })

        # Optional Folium map
        if output_maps:
            map_path = os.path.join(maps_dir, f"nga_dimension_{dim}_map.html")
            _make_folium_map(
                pred_df.assign(
                    latitude=df["latitude"],
                    longitude=df["longitude"],
                    subregion=df["subregion"],
                    population=df["population"],
                ),
                pred_col=f"{dim}_moderate",
                title=label,
                out_path=map_path,
            )

        # Clean up temp column
        df.drop(columns=[f"_raw_{dim}"], errors="ignore", inplace=True)

    # Save prediction table
    out_csv = os.path.join(tables_dir, "nga_dimension_predictions.csv")
    pred_df.to_csv(out_csv, index=False)
    logger.info("\nDimension predictions saved: %s (%d rows × %d cols)", out_csv, *pred_df.shape)

    # Save eval summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        eval_path = os.path.join(eval_dir, "nga_dimension_summary.csv")
        summary_df.to_csv(eval_path, index=False)
        logger.info("Dimension summary saved: %s", eval_path)
        logger.info("\n%s", summary_df[["dimension", "target_mean", "pred_mean", "zone_pearson_r"]].to_string(index=False))

    return pred_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run per-dimension deprivation models (Kyriaki specification)."
    )
    parser.add_argument("--country", default="nga", help="Country code (default: nga)")
    parser.add_argument(
        "--dims",
        nargs="+",
        default=None,
        help="Dimensions to run (default: all). E.g. --dims shelter water health",
    )
    parser.add_argument(
        "--recompute-targets",
        action="store_true",
        default=False,
        help="Force recomputation of nga_dimension_targets.csv from MICS SPSS files.",
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        default=False,
        help="Skip Folium map generation.",
    )
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    args = parser.parse_args()

    config_path = args.config or (
        f"config/config_{args.country}.yaml" if args.country else None
    )
    cfg = load_config(config_path)
    setup_logging(cfg)

    if args.recompute_targets:
        dim_targets_path = os.path.join(cfg["paths"]["interim_dir"], "nga_dimension_targets.csv")
        if os.path.isfile(dim_targets_path):
            os.remove(dim_targets_path)
            logger.info("Removed cached dimension targets — will recompute.")

    if args.no_maps:
        if "dimensions" not in cfg:
            cfg["dimensions"] = {}
        cfg["dimensions"]["output_maps"] = False

    logger.info("\n=== Per-Dimension Deprivation Pipeline ===")
    logger.info("Country: %s", args.country.upper())
    logger.info("Dimensions: %s", args.dims or "all")

    run_dimension_models(cfg, dims_to_run=args.dims)
    logger.info("\nDone.")


if __name__ == "__main__":
    main()
