"""
UNICEF × RBC Borealis AI — Child Deprivation Disaggregation Pipeline
=====================================================================

Main entry point for the full research pipeline.

Phases executed:
  1. Data pipeline (Steps 01–05) — produces clean modeling table
  2. Baselines — uniform allocation and RWI redistribution
  3. ML models — Ridge regression and (optional) GBM
  4. Evaluation — comparative metrics and report
  5. Outputs — save predictions, maps, evaluation tables

Usage
-----
    python main.py                          # Run full pipeline (Jamaica default)
    python main.py --country nga            # Run Nigeria pipeline
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
        description="Child Deprivation Disaggregation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Country code (e.g., 'nga' for Nigeria). Loads config/config_{country}.yaml.",
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
        "--skip-wsnn",
        action="store_true",
        default=False,
        help="Skip the weakly supervised neural network (requires: pip install torch).",
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
    """Apply uniform, heuristic, and RWI baselines."""
    from src.baselines.uniform import run as run_uniform
    from src.baselines.heuristic import run as run_heuristic
    from src.baselines.rwi_redistribution import run as run_rwi

    logger.info("\n[Phase 3: Baselines]")

    logger.info("--- Applying Uniform Baseline ---")
    df = run_uniform(cfg, df)

    logger.info("--- Applying Heuristic Baseline ---")
    df = run_heuristic(cfg, df)

    logger.info("--- Applying RWI Redistribution Baseline ---")
    df = run_rwi(cfg, df)

    return df


def phase_models(
    cfg: dict,
    df: pd.DataFrame,
    skip_gbm: bool = False,
    skip_gam: bool = False,
    skip_wsnn: bool = False,
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

    # Weakly Supervised Neural Network (optional - requires PyTorch)
    wsnn_model = None
    if not skip_wsnn:
        logger.info("--- Fitting Weakly Supervised Neural Network (WSNN) ---")
        try:
            from src.models.weakly_supervised_nn import run as run_wsnn
            df, wsnn_model = run_wsnn(cfg, df)
        except ImportError as e:
            logger.warning(
                "WSNN model skipped — PyTorch not installed: %s. "
                "Install with: pip install torch",
                e,
            )
        except Exception as e:
            logger.error("WSNN model failed: %s", e, exc_info=True)
            logger.warning("Continuing without WSNN predictions.")

    return df, ridge_model, gam_model, gbm_model, fi_df


def phase_eval(cfg: dict, df: pd.DataFrame, skip_gbm: bool = False, skip_gam: bool = False, skip_wsnn: bool = False) -> dict:
    """Run evaluation and return results dict."""
    from src.evaluation.metrics import (
        evaluate_all, format_eval_report,
        paired_significance_tests, rwi_uncertainty_analysis,
    )
    from src.evaluation.zone_cv import leave_one_zone_out

    logger.info("\n[Phase 6: Evaluation]")
    results = evaluate_all(df, cfg)
    report = format_eval_report(results)

    logger.info("\n=== Evaluation Report ===")
    logger.info("\n%s", report.to_string())

    # Leave-one-zone-out cross-validation
    eval_dir = cfg["paths"]["eval_dir"]
    os.makedirs(eval_dir, exist_ok=True)

    logger.info("\n--- Leave-One-Zone-Out CV ---")
    lozo_df = leave_one_zone_out(df, cfg, skip_gbm=skip_gbm, skip_gam=skip_gam, skip_wsnn=skip_wsnn)
    if len(lozo_df) > 0:
        lozo_path = os.path.join(eval_dir, "lozo_evaluation.csv")
        lozo_df.to_csv(lozo_path, index=False)
        logger.info("LOZO evaluation saved to: %s", lozo_path)

    # Statistical significance tests
    logger.info("\n--- Significance Tests ---")
    sig_df = paired_significance_tests(df, cfg)
    if len(sig_df) > 0:
        sig_path = os.path.join(eval_dir, "significance_tests.csv")
        sig_df.to_csv(sig_path, index=False)
        logger.info("Significance tests saved to: %s", sig_path)

    # RWI uncertainty analysis
    logger.info("\n--- RWI Uncertainty Analysis ---")
    rwi_unc_df = rwi_uncertainty_analysis(df)
    if len(rwi_unc_df) > 0:
        rwi_unc_path = os.path.join(eval_dir, "rwi_uncertainty_analysis.csv")
        rwi_unc_df.to_csv(rwi_unc_path, index=False)
        logger.info("RWI uncertainty analysis saved to: %s", rwi_unc_path)

    # Two-level cross-validation (if enabled in config)
    two_level_cfg = cfg.get("evaluation", {}).get("two_level_cv", {})
    if two_level_cfg.get("enabled", False):
        logger.info("\n--- Two-Level Cross-Validation ---")
        try:
            from src.evaluation.two_level_cv import two_level_cross_validation
            max_folds = two_level_cfg.get("max_folds", 10)
            tlcv_df = two_level_cross_validation(df, cfg, max_folds=max_folds)
            if len(tlcv_df) > 0:
                tlcv_path = os.path.join(eval_dir, "two_level_cv.csv")
                tlcv_df.to_csv(tlcv_path, index=False)
                logger.info("Two-level CV saved to: %s", tlcv_path)
        except Exception as e:
            logger.error("Two-level CV failed: %s", e, exc_info=True)

    # --- Hierarchical Cross-Level Validation ---
    hcv_cfg = cfg.get("evaluation", {}).get("hierarchical_cv", {})
    if hcv_cfg.get("enabled", False):
        logger.info("\n--- Hierarchical Cross-Level Validation ---")
        try:
            from src.evaluation.hierarchical_cv import hierarchical_validation
            interim_dir = cfg["paths"]["interim_dir"]
            hcv_df = hierarchical_validation(df, cfg, interim_dir=interim_dir, eval_dir=eval_dir)
            if len(hcv_df) > 0:
                results["hierarchical_cv"] = hcv_df
        except Exception as e:
            logger.error("Hierarchical CV failed: %s", e, exc_info=True)

    return results, report


def phase_outputs(
    cfg: dict,
    df: pd.DataFrame,
    eval_results: dict,
    eval_report: pd.DataFrame,
    fi_df: pd.DataFrame = None,
    ridge_model=None,
    gbm_model=None,
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
        if any(col.startswith(p) for p in ["ridge_", "gam_", "gbm_", "heuristic_", "wsnn_"]):
            pred_cols.append(col)

    pred_cols = [c for c in pred_cols if c in df.columns]
    pred_table = df[pred_cols].copy()

    # Ridge per-cell explainability: CSV + theme/popup columns merged for Folium/GeoJSON
    try:
        from src.outputs.prediction_breakdown import (
            export_ridge_breakdown,
            merge_breakdown_into_pred_table,
            maybe_export_gbm_shap,
        )
        _, brk_df = export_ridge_breakdown(cfg, df, ridge_model)
        pred_table = merge_breakdown_into_pred_table(pred_table, brk_df)
        maybe_export_gbm_shap(cfg, df, gbm_model)
    except Exception as e:
        logger.warning("Prediction breakdown / SHAP export failed: %s", e)

    output_prefix = cfg.get("country", {}).get("output_prefix", "jam")
    pred_parquet = os.path.join(tables_dir, f"{output_prefix}_predictions.parquet")
    pred_csv = os.path.join(tables_dir, f"{output_prefix}_predictions.csv")
    pred_table.to_parquet(pred_parquet, index=False)
    pred_table.to_csv(pred_csv, index=False)
    logger.info("Predictions saved to: %s  (%d rows)", pred_parquet, len(pred_table))

    consolidated_path = os.path.join(tables_dir, f"{output_prefix}_full_consolidated.parquet")
    df.to_parquet(consolidated_path, index=False)
    logger.info(
        "Full consolidated table (features + predictions): %s — %d rows × %d cols",
        consolidated_path,
        len(df),
        len(df.columns),
    )

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
        geojson_path = os.path.join(maps_dir, f"{output_prefix}_predictions.geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")
        logger.info("Map-ready GeoJSON saved to: %s", geojson_path)
    except Exception as e:
        logger.warning("Could not save GeoJSON: %s", e)

    # ------------------------------------------------------------------
    # Interactive Folium map (optional — requires folium)
    # ------------------------------------------------------------------
    try:
        import folium
        from folium.plugins import MarkerCluster, FastMarkerCluster
        import branca.colormap as cm

        map_cfg = cfg.get("maps", {})
        use_cluster = map_cfg.get("use_folium_cluster", False)
        cluster_threshold = int(map_cfg.get("folium_cluster_threshold", 10_000))
        max_cells_full = int(map_cfg.get("folium_max_cells_full", 0))   # 0 = no cap
        sample_cells = int(map_cfg.get("folium_sample_cells", 5_000))   # sampled map cells

        map_df = pred_table[pred_table["moderate_prevalence"].notna()].copy()
        n_cells = len(map_df)

        # Auto-tune opacity and radius for dense maps
        if n_cells > 50_000:
            fill_opacity = 0.35
            marker_radius = 3
        elif n_cells > 20_000:
            fill_opacity = 0.50
            marker_radius = 3
        else:
            fill_opacity = 0.70
            marker_radius = 4

        # Auto-enable clustering if above threshold (unless config explicitly sets it False)
        effective_cluster = use_cluster or (n_cells >= cluster_threshold and map_cfg.get("use_folium_cluster") is None)

        center_lat = map_df["latitude"].mean()
        center_lon = map_df["longitude"].mean()

        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                          tiles="CartoDB positron")

        # Pick the best available prediction column
        col_priority = ["gbm_moderate", "gam_moderate", "ridge_moderate",
                        "rwi_moderate", "uniform_moderate"]
        plot_col = next((c for c in col_priority if c in map_df.columns), None)

        def _make_popup_html(row, plot_col_val):
            explain = row.get("ridge_bdg_popup")
            base = (
                f"<b>{row.get('parish_name', '')} / {row.get('subregion', '')}</b><br>"
                f"RWI: {row.get('rwi', 'N/A'):.2f}<br>"
                f"Pop: {row.get('population', 'N/A'):.0f}<br>"
                f"Moderate poverty: {plot_col_val:.1f}%"
            )
            if explain and isinstance(explain, str) and explain.strip():
                return base + explain, 360
            return base, 220

        def _folium_legend_html(plot_col_name: str, n: int, is_sample: bool) -> str:
            note = f"<br><i style='color:#888;font-size:10px;'>{n:,} cells shown" + (" (sampled)" if is_sample else "") + "</i>"
            return f"""
            <div style="position:fixed;top:12px;left:60px;z-index:1000;
                        background:white;padding:10px 14px;border-radius:8px;
                        box-shadow:2px 2px 8px rgba(0,0,0,.3);
                        font-family:Arial,sans-serif;max-width:310px;font-size:12px;">
              <b style="font-size:13px;">Nigeria Child Deprivation — Cell Level</b><br>
              <span style="color:#555;">
                Each <b>circle = 1 grid cell</b> (~1 km²).<br>
                <b>Colour</b> = {plot_col_name.replace('_',' ')} (%)
                &nbsp; <b style="color:#2166ac;">▊</b> low &rarr;
                       <b style="color:#d73027;">▊</b> high<br>
                <b>Click</b> a circle for values + Ridge explain block.<br>
                For national overview use <code>nga_comparison_map.html</code>.{note}
              </span>
            </div>"""

        if plot_col:
            vmin, vmax = map_df[plot_col].quantile(0.02), map_df[plot_col].quantile(0.98)
            colormap = cm.LinearColormap(
                ["#2166ac", "#f7f7f7", "#d73027"],
                vmin=vmin, vmax=vmax,
                caption=f"{plot_col.replace('_', ' ').title()} (%)",
            )
            colormap.add_to(fmap)

            # Optionally cap rows (full map)
            draw_df = map_df
            if max_cells_full > 0 and n_cells > max_cells_full:
                draw_df = map_df.sample(n=max_cells_full, random_state=42)
                logger.info("Folium: capped to %d cells (folium_max_cells_full).", max_cells_full)

            if effective_cluster:
                layer = MarkerCluster(name="Cells")
                layer.add_to(fmap)
                dest = layer
            else:
                dest = fmap

            for _, row in draw_df.iterrows():
                val = row[plot_col]
                if pd.isna(val):
                    continue
                pop_html, w = _make_popup_html(row, val)
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=marker_radius,
                    color=None,
                    fill=True,
                    fill_color=colormap(val),
                    fill_opacity=fill_opacity,
                    popup=folium.Popup(pop_html, max_width=w),
                ).add_to(dest)

            fmap.get_root().html.add_child(
                folium.Element(_folium_legend_html(plot_col, len(draw_df), max_cells_full > 0 and n_cells > max_cells_full))
            )

        html_path = os.path.join(maps_dir, f"{output_prefix}_predictions_map.html")
        fmap.save(html_path)
        logger.info("Folium interactive map saved to: %s (n=%d, cluster=%s, opacity=%.2f)",
                    html_path, len(map_df), effective_cluster, fill_opacity)

        # ---- Sampled map: lighter file for quick demos ----------------
        if plot_col and n_cells > sample_cells:
            fmap_s = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                                tiles="CartoDB positron")
            colormap.add_to(fmap_s)
            strat_col = "subregion" if "subregion" in map_df.columns else None
            if strat_col:
                per_state = max(1, sample_cells // map_df[strat_col].nunique())
                sampled = (
                    map_df.groupby(strat_col, group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), per_state), random_state=42))
                    .reset_index(drop=True)
                )
            else:
                sampled = map_df.sample(n=min(sample_cells, n_cells), random_state=42)
            for _, row in sampled.iterrows():
                val = row[plot_col]
                if pd.isna(val):
                    continue
                pop_html, w = _make_popup_html(row, val)
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=4,
                    color=None,
                    fill=True,
                    fill_color=colormap(val),
                    fill_opacity=0.70,
                    popup=folium.Popup(pop_html, max_width=w),
                ).add_to(fmap_s)
            fmap_s.get_root().html.add_child(
                folium.Element(_folium_legend_html(plot_col, len(sampled), True))
            )
            sample_html_path = os.path.join(maps_dir, f"{output_prefix}_predictions_map_sample.html")
            fmap_s.save(sample_html_path)
            logger.info("Folium sampled map saved to: %s (%d cells)", sample_html_path, len(sampled))

    except ImportError:
        logger.info("folium not installed — skipping interactive map. "
                    "Install with: pip install folium branca")
    except Exception as e:
        logger.warning("Could not save Folium map: %s", e)

    # ------------------------------------------------------------------
    # Uncertainty map (CI width) — requires folium + CI columns
    # ------------------------------------------------------------------
    try:
        import folium
        import branca.colormap as cm

        # Find the best available CI columns
        ci_priority = [
            ("gbm_moderate_lower", "gbm_moderate_upper", "gbm_moderate"),
            ("gam_moderate_lower", "gam_moderate_upper", "gam_moderate"),
            ("ridge_moderate_lower", "ridge_moderate_upper", "ridge_moderate"),
        ]
        ci_cols = None
        for lower_c, upper_c, pred_c in ci_priority:
            if lower_c in pred_table.columns and upper_c in pred_table.columns:
                ci_cols = (lower_c, upper_c, pred_c)
                break

        if ci_cols is not None:
            lower_c, upper_c, pred_c = ci_cols
            unc_df = pred_table[
                pred_table["moderate_prevalence"].notna()
                & pred_table[lower_c].notna()
            ].copy()
            unc_df["ci_width"] = unc_df[upper_c] - unc_df[lower_c]

            n_unc = len(unc_df)
            u_opacity = 0.35 if n_unc > 50_000 else (0.50 if n_unc > 20_000 else 0.70)
            u_radius = 3 if n_unc > 20_000 else 4

            center_lat = unc_df["latitude"].mean()
            center_lon = unc_df["longitude"].mean()

            umap = folium.Map(
                location=[center_lat, center_lon], zoom_start=6,
                tiles="CartoDB positron",
            )
            umap.get_root().html.add_child(folium.Element(f"""
            <div style="position:fixed;top:12px;left:60px;z-index:1000;
                        background:white;padding:10px 14px;border-radius:8px;
                        box-shadow:2px 2px 8px rgba(0,0,0,.3);
                        font-family:Arial,sans-serif;max-width:300px;font-size:12px;">
              <b style="font-size:13px;">Prediction Uncertainty — Cell Level</b><br>
              <span style="color:#555;">
                Each <b>circle = 1 grid cell</b>.<br>
                <b>Colour</b> = 90% CI width (pp) &nbsp;
                <b style="color:#2c7bb6;">▊</b> narrow &rarr;
                <b style="color:#d7191c;">▊</b> wide<br>
                <b>Click</b> for CI bounds + Ridge explain.<br>
                <i style="font-size:10px;">{n_unc:,} cells shown</i>
              </span>
            </div>"""))

            vmin = unc_df["ci_width"].quantile(0.02)
            vmax = unc_df["ci_width"].quantile(0.98)
            colormap = cm.LinearColormap(
                ["#2166ac", "#f7f7f7", "#d73027"],
                vmin=vmin, vmax=vmax,
                caption="90% CI Width (pp)",
            )
            colormap.add_to(umap)

            for _, row in unc_df.iterrows():
                w = row["ci_width"]
                if pd.isna(w):
                    continue
                uexplain = row.get("ridge_bdg_popup")
                if (
                    uexplain is not None
                    and isinstance(uexplain, str)
                    and uexplain.strip()
                ):
                    uhtml = (
                        f"<b>{row.get('parish_name', '')} / {row.get('subregion', '')}</b><br>"
                        f"CI width: {w:.2f} pp<br>"
                        f"Lower: {row[lower_c]:.1f}%<br>"
                        f"Upper: {row[upper_c]:.1f}%<br>"
                        f"Prediction: {row.get(pred_c, 'N/A'):.1f}%"
                        f"{uexplain}"
                    )
                    uw = 360
                else:
                    uhtml = (
                        f"<b>{row.get('parish_name', '')} / {row.get('subregion', '')}</b><br>"
                        f"CI width: {w:.2f} pp<br>"
                        f"Lower: {row[lower_c]:.1f}%<br>"
                        f"Upper: {row[upper_c]:.1f}%<br>"
                        f"Prediction: {row.get(pred_c, 'N/A'):.1f}%"
                    )
                    uw = 220
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=u_radius,
                    color=None,
                    fill=True,
                    fill_color=colormap(np.clip(w, vmin, vmax)),
                    fill_opacity=u_opacity,
                    popup=folium.Popup(uhtml, max_width=uw),
                ).add_to(umap)

            unc_html_path = os.path.join(maps_dir, f"{output_prefix}_uncertainty_map.html")
            umap.save(unc_html_path)
            logger.info("Uncertainty map saved to: %s", unc_html_path)
        else:
            logger.info("No CI columns found — skipping uncertainty map.")

    except ImportError:
        logger.info("folium not installed — skipping uncertainty map.")
    except Exception as e:
        logger.warning("Could not save uncertainty map: %s", e)

    # ------------------------------------------------------------------
    # LGA-level aggregation (Nigeria only — GADM ADM2)
    # ------------------------------------------------------------------
    country_code = cfg.get("country", {}).get("code", "")
    if country_code == "NGA":
        try:
            from src.outputs.lga_aggregation import aggregate_to_lga
            lga_csv, lga_geojson = aggregate_to_lga(cfg, pred_table)
            if lga_csv:
                logger.info("LGA-level predictions saved to: %s", lga_csv)
            if lga_geojson:
                logger.info("LGA GeoJSON saved to: %s", lga_geojson)
        except Exception as e:
            logger.warning("LGA aggregation failed: %s", e)

    logger.info("All outputs saved to: %s", cfg["paths"]["outputs_dir"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve config path: --config takes priority, then --country, then default
    config_path = args.config
    if config_path is None and args.country:
        from src.utils.config_loader import find_project_root
        project_root = find_project_root()
        config_path = os.path.join(
            project_root, "config", f"config_{args.country.lower()}.yaml"
        )

    cfg = load_config(config_path)
    setup_logging(cfg)

    country_name = cfg.get("country", {}).get("name", "Jamaica")

    logger.info("=" * 60)
    logger.info("UNICEF × RBC Borealis AI — %s Child Deprivation Pipeline", country_name)
    logger.info("Phase: %s | Force rerun: %s | Skip GBM: %s | Skip GAM: %s",
                args.phase, args.force_rerun, args.skip_gbm, args.skip_gam)
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
    df, ridge_model, gam_model, gbm_model, fi_df = phase_models(
        cfg, df, skip_gbm=args.skip_gbm, skip_gam=args.skip_gam, skip_wsnn=args.skip_wsnn
    )

    if args.phase == "models":
        logger.info("Stopping after models phase.")
        return

    # Phase 5: Verification of reconciliation
    logger.info("\n[Phase 5: Reconciliation Verification]")
    from src.reconciliation.admin_reconcile import verify_reconciliation

    verification_pairs = [
        ("uniform_moderate", "moderate_prevalence"),
        ("heuristic_moderate", "moderate_prevalence"),
        ("rwi_moderate", "moderate_prevalence"),
        ("ridge_moderate", "moderate_prevalence"),
        ("gam_moderate", "moderate_prevalence"),
        ("gbm_moderate", "moderate_prevalence"),
        ("wsnn_moderate", "moderate_prevalence"),
        # Depth metrics
        ("uniform_moderate_depth", "moderate_depth"),
        ("heuristic_moderate_depth", "moderate_depth"),
        ("rwi_moderate_depth", "moderate_depth"),
        ("ridge_moderate_depth", "moderate_depth"),
        ("gam_moderate_depth", "moderate_depth"),
        ("gbm_moderate_depth", "moderate_depth"),
        ("wsnn_moderate_depth", "moderate_depth"),
    ]
    for col, target_col in verification_pairs:
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

    if args.phase == "eval" or args.phase == "all":
        # Phase 6: Evaluation
        eval_results, eval_report = phase_eval(cfg, df, skip_gbm=args.skip_gbm, skip_gam=args.skip_gam, skip_wsnn=args.skip_wsnn)

        # Phase 7: Outputs
        phase_outputs(
            cfg, df, eval_results, eval_report,
            fi_df=fi_df, ridge_model=ridge_model, gbm_model=gbm_model,
        )

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
