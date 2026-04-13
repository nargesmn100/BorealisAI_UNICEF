"""
UNICEF × RBC Borealis AI — Jamaica Child Deprivation Pipeline
==============================================================

Main entry point for the full research pipeline.

Phases executed:
  1. Data pipeline (Steps 01–05) — produces clean modeling table
  2. Baselines — uniform allocation and RWI redistribution
  3. ML models — Ridge regression and (optional) GBM
  4. Evaluation — comparative metrics and report
  5. Outputs — save predictions, maps, evaluation tables

Usage
-----
    python main.py                          # Run full pipeline
    python main.py --force-rerun            # Re-run all steps
    python main.py --skip-gbm              # Skip gradient boosting
    python main.py --phase data            # Only run data pipeline
    python main.py --phase baselines       # Data + baselines
    python main.py --phase models          # Data + baselines + models
    python main.py --phase eval            # Full pipeline through eval

Design notes
------------
- Intermediate files are cached so individual phases can be re-run.
- All outputs go to data/outputs/{tables,maps,eval}/.
- Logs are written to stdout.  Redirect to file if needed.
- The pipeline raises loudly on bad joins, CRS mismatches, or missing files.

Important framing
-----------------
This system is a RESEARCH TOOL, not an official statistics system.
Outputs are NOT official poverty estimates.  They represent one possible
spatial disaggregation consistent with official totals.
"""

import argparse
import logging
import os
import sys

import pandas as pd
import numpy as np

from src.utils.config_loader import load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jamaica Child Deprivation Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        default=False,
        help="Re-run all pipeline steps even if cached outputs exist.",
    )
    parser.add_argument(
        "--skip-gbm",
        action="store_true",
        default=False,
        help="Skip the gradient boosting model (faster runs for debugging).",
    )
    parser.add_argument(
        "--skip-gam",
        action="store_true",
        default=False,
        help="Skip the GAM model (requires: pip install pygam).",
    )
    parser.add_argument(
        "--skip-ws",
        action="store_true",
        default=False,
        help="Skip weakly supervised models (WeaklySupervisedLinear + MLP).",
    )
    parser.add_argument(
        "--skip-region-split",
        action="store_true",
        default=False,
        help="Skip leave-one-zone-out cross-validation (faster runs).",
    )
    parser.add_argument(
        "--phase",
        choices=["data", "baselines", "models", "eval", "all"],
        default="all",
        help="Pipeline phase to run up to (default: all).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def phase_data(cfg: dict, force_rerun: bool) -> pd.DataFrame:
    """Run the data pipeline and return the modeling table."""
    from src.pipeline.run_pipeline import run_data_pipeline
    return run_data_pipeline(cfg, force_rerun=force_rerun)


def phase_baselines(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Apply uniform and RWI baselines."""
    from src.baselines.uniform import run as run_uniform
    from src.baselines.rwi_redistribution import run as run_rwi

    logger.info("\n[Phase 3: Baselines]")

    logger.info("--- Applying Uniform Baseline ---")
    df = run_uniform(cfg, df)

    logger.info("--- Applying RWI Redistribution Baseline ---")
    df = run_rwi(cfg, df)

    return df


def phase_models(
    cfg: dict,
    df: pd.DataFrame,
    skip_gbm: bool = False,
    skip_gam: bool = False,
    skip_ws: bool = False,
) -> tuple:
    """Train and apply ML models."""
    from src.models.ridge_model import run as run_ridge

    logger.info("\n[Phase 4: ML Models]")

    # Ridge regression
    logger.info("--- Fitting Ridge Regression ---")
    df, ridge_model = run_ridge(cfg, df)

    # GAM (optional — requires pygam)
    gam_model = None
    if not skip_gam:
        logger.info("--- Fitting Generalised Additive Model (GAM) ---")
        try:
            from src.models.gam_model import run as run_gam
            df, gam_model = run_gam(cfg, df)
        except ImportError as e:
            logger.warning(
                "GAM model skipped — pygam not installed: %s. "
                "Install with: pip install pygam",
                e,
            )
        except Exception as e:
            logger.error("GAM model failed: %s", e, exc_info=True)
            logger.warning("Continuing without GAM predictions.")

    # GBM (optional)
    gbm_model = None
    fi_df = None
    if not skip_gbm:
        logger.info("--- Fitting Gradient Boosted Trees ---")
        try:
            from src.models.gbm_model import run as run_gbm
            df, gbm_model, fi_df = run_gbm(cfg, df)
        except ImportError as e:
            logger.warning(
                "GBM model skipped — dependency not available: %s. "
                "Install with: pip install lightgbm",
                e,
            )
        except Exception as e:
            logger.error("GBM model failed with error: %s", e, exc_info=True)
            logger.warning("Continuing without GBM predictions.")

    # Weakly supervised models — the principled approach from the problem statement:
    # train with zone-level aggregation loss instead of cell-level surrogate loss.
    ws_linear = ws_mlp = ws_linear_imp = ws_mlp_imp = None
    if not skip_ws:
        logger.info("--- Fitting Weakly Supervised Models ---")
        try:
            from src.models.weakly_supervised_model import run as run_ws
            df, ws_linear, ws_mlp, ws_linear_imp, ws_mlp_imp = run_ws(cfg, df)
        except Exception as e:
            logger.error("Weakly supervised models failed: %s", e, exc_info=True)
            logger.warning("Continuing without weakly supervised predictions.")

    return df, ridge_model, gam_model, gbm_model, fi_df, ws_linear, ws_mlp, ws_linear_imp, ws_mlp_imp


def phase_eval(cfg: dict, df: pd.DataFrame) -> dict:
    """Run evaluation and return results dict."""
    from src.evaluation.metrics import evaluate_all, format_eval_report

    logger.info("\n[Phase 6: Evaluation]")
    results = evaluate_all(df, cfg)
    report = format_eval_report(results)

    logger.info("\n=== Evaluation Report ===")
    logger.info("\n%s", report.to_string())

    return results, report


def phase_region_split_eval(cfg: dict, df: pd.DataFrame) -> dict:
    """
    Run leave-one-zone-out and within-zone holdout evaluation.

    This tests whether the weakly supervised models can generalise to
    zones they have never seen during training — the core generalisation
    test from the problem statement.
    """
    from src.evaluation.region_split import run_region_split_evaluation

    logger.info("\n[Phase 6b: Region-Split Generalisation Evaluation]")
    return run_region_split_evaluation(df, cfg)


def phase_outputs(
    cfg: dict,
    df: pd.DataFrame,
    eval_results: dict,
    eval_report: pd.DataFrame,
    fi_df: pd.DataFrame = None,
    ws_linear_imp: pd.DataFrame = None,
    ws_mlp_imp: pd.DataFrame = None,
    region_split_results: dict = None,
) -> None:
    """Save all outputs to disk."""
    logger.info("\n[Phase 7: Outputs]")

    tables_dir = cfg["paths"]["tables_dir"]
    maps_dir = cfg["paths"]["maps_dir"]
    eval_dir = cfg["paths"]["eval_dir"]

    for d in [tables_dir, maps_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Full prediction table (Parquet + CSV)
    # ------------------------------------------------------------------
    pred_cols = [
        "cell_id", "latitude", "longitude", "rwi", "population",
        "is_urban", "smod_class", "smod_label",
        "travel_time_cities", "travel_time_50k",
        "parish_name", "subregion",
        "moderate_prevalence", "severe_prevalence",
        "uniform_moderate", "uniform_severe",
        "rwi_moderate", "rwi_severe",
    ]

    # Add model prediction columns if available
    for col in df.columns:
        if any(col.startswith(p) for p in ["ridge_", "gam_", "gbm_", "ws_"]):
            pred_cols.append(col)

    pred_cols = [c for c in pred_cols if c in df.columns]
    pred_table = df[pred_cols].copy()

    pred_parquet = os.path.join(tables_dir, "jam_predictions.parquet")
    pred_csv = os.path.join(tables_dir, "jam_predictions.csv")
    pred_table.to_parquet(pred_parquet, index=False)
    pred_table.to_csv(pred_csv, index=False)
    logger.info("Predictions saved to: %s  (%d rows)", pred_parquet, len(pred_table))

    # ------------------------------------------------------------------
    # Evaluation report
    # ------------------------------------------------------------------
    eval_csv = os.path.join(eval_dir, "evaluation_summary.csv")
    eval_report.to_csv(eval_csv)
    logger.info("Evaluation summary saved to: %s", eval_csv)

    # Admin detail tables
    for method, res in eval_results.items():
        if "admin_detail" in res and isinstance(res["admin_detail"], pd.DataFrame):
            detail_path = os.path.join(eval_dir, f"admin_detail_{method}.csv")
            res["admin_detail"].to_csv(detail_path, index=False)

    # ------------------------------------------------------------------
    # GBM feature importance (if available)
    # ------------------------------------------------------------------
    if fi_df is not None and isinstance(fi_df, pd.DataFrame) and len(fi_df) > 0:
        fi_path = os.path.join(eval_dir, "gbm_feature_importances.csv")
        fi_df.to_csv(fi_path, index=False)
        logger.info("GBM feature importances saved to: %s", fi_path)

    # ------------------------------------------------------------------
    # Weakly supervised model feature importances (permutation-based)
    # ------------------------------------------------------------------
    if ws_linear_imp is not None and isinstance(ws_linear_imp, pd.DataFrame):
        p = os.path.join(eval_dir, "ws_linear_permutation_importance.csv")
        ws_linear_imp.to_csv(p, index=False)
        logger.info("WS-Linear permutation importances saved to: %s", p)

    if ws_mlp_imp is not None and isinstance(ws_mlp_imp, pd.DataFrame):
        p = os.path.join(eval_dir, "ws_mlp_permutation_importance.csv")
        ws_mlp_imp.to_csv(p, index=False)
        logger.info("WS-MLP permutation importances saved to: %s", p)

    # ------------------------------------------------------------------
    # Region-split evaluation results (LOSO CV + within-zone holdout)
    # ------------------------------------------------------------------
    if region_split_results:
        for key, df_res in region_split_results.items():
            if isinstance(df_res, pd.DataFrame) and len(df_res) > 0:
                p = os.path.join(eval_dir, f"region_split_{key}.csv")
                df_res.to_csv(p, index=False)
                logger.info("Region split result '%s' saved to: %s", key, p)

    # ------------------------------------------------------------------
    # Map-ready GeoJSON (for quick visualisation)
    # ------------------------------------------------------------------
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        modeling_subset = pred_table[pred_table["moderate_prevalence"].notna()].copy()
        geometry = gpd.points_from_xy(
            modeling_subset["longitude"], modeling_subset["latitude"]
        )
        gdf = gpd.GeoDataFrame(modeling_subset, geometry=geometry, crs="EPSG:4326")
        geojson_path = os.path.join(maps_dir, "jam_predictions.geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")
        logger.info("Map-ready GeoJSON saved to: %s", geojson_path)
    except Exception as e:
        logger.warning("Could not save GeoJSON: %s", e)

    # ------------------------------------------------------------------
    # Interactive Folium map (optional — requires folium)
    # ------------------------------------------------------------------
    try:
        import folium
        from folium.plugins import MarkerCluster
        import branca.colormap as cm

        map_df = pred_table[pred_table["moderate_prevalence"].notna()].copy()
        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=9,
                          tiles="CartoDB positron")

        # Pick the best available prediction column
        col_priority = ["gbm_moderate", "gam_moderate", "ridge_moderate",
                        "rwi_moderate", "uniform_moderate"]
        plot_col = next((c for c in col_priority if c in map_df.columns), None)

        if plot_col:
            vmin, vmax = map_df[plot_col].quantile(0.02), map_df[plot_col].quantile(0.98)
            colormap = cm.LinearColormap(
                ["#2166ac", "#f7f7f7", "#d73027"],
                vmin=vmin, vmax=vmax,
                caption=f"{plot_col.replace('_', ' ').title()} (%)",
            )
            colormap.add_to(fmap)

            for _, row in map_df.iterrows():
                val = row[plot_col]
                if pd.isna(val):
                    continue
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=4,
                    color=None,
                    fill=True,
                    fill_color=colormap(val),
                    fill_opacity=0.75,
                    popup=folium.Popup(
                        f"<b>{row.get('parish_name', '')} / {row.get('subregion', '')}</b><br>"
                        f"RWI: {row.get('rwi', 'N/A'):.2f}<br>"
                        f"Pop: {row.get('population', 'N/A'):.0f}<br>"
                        f"Moderate poverty: {val:.1f}%",
                        max_width=220,
                    ),
                ).add_to(fmap)

        html_path = os.path.join(maps_dir, "jam_predictions_map.html")
        fmap.save(html_path)
        logger.info("Folium interactive map saved to: %s", html_path)
    except ImportError:
        logger.info("folium not installed — skipping interactive map. "
                    "Install with: pip install folium branca")
    except Exception as e:
        logger.warning("Could not save Folium map: %s", e)

    logger.info("All outputs saved to: %s", cfg["paths"]["outputs_dir"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    logger.info("=" * 60)
    logger.info("UNICEF × RBC Borealis AI — Jamaica Child Deprivation Pipeline")
    logger.info(
        "Phase: %s | Force rerun: %s | Skip GBM: %s | Skip GAM: %s | "
        "Skip WS: %s | Skip region-split: %s",
        args.phase, args.force_rerun, args.skip_gbm, args.skip_gam,
        args.skip_ws, args.skip_region_split,
    )
    logger.info("=" * 60)

    # Phase 1+2: Data pipeline
    df = phase_data(cfg, force_rerun=args.force_rerun)

    if args.phase == "data":
        logger.info("Stopping after data pipeline phase.")
        return

    # Phase 3: Baselines
    df = phase_baselines(cfg, df)

    if args.phase == "baselines":
        logger.info("Stopping after baselines phase.")
        return

    # Phase 4: ML Models
    df, ridge_model, gam_model, gbm_model, fi_df, ws_linear, ws_mlp, ws_linear_imp, ws_mlp_imp = phase_models(
        cfg, df, skip_gbm=args.skip_gbm, skip_gam=args.skip_gam, skip_ws=args.skip_ws,
    )

    if args.phase == "models":
        logger.info("Stopping after models phase.")
        return

    # Phase 5: Verification of reconciliation
    logger.info("\n[Phase 5: Reconciliation Verification]")
    from src.reconciliation.admin_reconcile import verify_reconciliation

    reconcile_cols = [
        ("uniform_moderate", "moderate_prevalence"),
        ("rwi_moderate", "moderate_prevalence"),
        ("ridge_moderate", "moderate_prevalence"),
        ("gam_moderate", "moderate_prevalence"),
        ("gbm_moderate", "moderate_prevalence"),
        ("ws_linear_moderate", "moderate_prevalence"),
        ("ws_mlp_moderate", "moderate_prevalence"),
    ]
    for col, target_col in reconcile_cols:
        if col in df.columns:
            ok = verify_reconciliation(
                df, col, target_col, "subregion", "population"
            )
            if not ok:
                logger.error(
                    "Reconciliation verification FAILED for '%s'. "
                    "Check pipeline for bugs.",
                    col,
                )

    region_split_results = None
    if args.phase == "eval" or args.phase == "all":
        # Phase 6: Evaluation
        eval_results, eval_report = phase_eval(cfg, df)

        # Phase 6b: Region-split evaluation (LOSO CV)
        if not args.skip_region_split and not args.skip_ws:
            region_split_results = phase_region_split_eval(cfg, df)
        else:
            logger.info("Skipping region-split evaluation.")

        # Phase 7: Outputs
        phase_outputs(
            cfg, df, eval_results, eval_report,
            fi_df=fi_df,
            ws_linear_imp=ws_linear_imp,
            ws_mlp_imp=ws_mlp_imp,
            region_split_results=region_split_results,
        )

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
