"""
DHS Auxiliary-Stack Sweep — Ridge training loss experiment.

Compares Ridge trained with:
  (a) DHS soft-label blend only (current default, dhs_aux_dhs_scale=0)
  (b) Stacked DHS auxiliary term at several scale values

Reports:
  - DHS external validation: Spearman ρ, MAE vs dhs_nearest_dep_index × 100
  - LOZO MAE across held-out states

Output
------
  Data/outputs/nga/eval/dhs_aux_stack_sweep.csv

Usage
-----
  python src/scripts/dhs_aux_sweep.py
  python src/scripts/dhs_aux_sweep.py --scales 0 0.1 0.25 0.5 1.0 2.0 --skip-lozo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

EVAL_DIR = ROOT / "Data/outputs/nga/eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dhs_external_metrics(df: pd.DataFrame) -> dict:
    """DHS GPS cluster-level validation (Spearman ρ, MAE)."""
    from scipy.stats import spearmanr, pearsonr
    mask = df["dhs_nearest_dep_index"].notna() & df["ridge_moderate"].notna()
    if mask.sum() < 10:
        return {"dhs_spearman": np.nan, "dhs_pearson": np.nan, "dhs_mae": np.nan, "dhs_n": int(mask.sum())}
    y_dhs = df.loc[mask, "dhs_nearest_dep_index"].values * 100.0
    y_pred = df.loc[mask, "ridge_moderate"].values
    sp = spearmanr(y_dhs, y_pred).statistic
    pr = pearsonr(y_dhs, y_pred)[0]
    mae = float(np.mean(np.abs(y_pred - y_dhs)))
    return {"dhs_spearman": round(sp, 4), "dhs_pearson": round(pr, 4),
            "dhs_mae": round(mae, 3), "dhs_n": int(mask.sum())}


def _lozo_mae(cfg: dict, df: pd.DataFrame) -> float | None:
    """Quick LOZO: mean |held-out zone state mean pred − truth| over all states."""
    from src.models.ridge_model import run as run_ridge
    zone_col = cfg["modeling"]["admin_zone_col"]
    zones = df[df["in_modeling_sample"].fillna(False)][zone_col].dropna().unique()
    if len(zones) < 2:
        return None
    errors = []
    for zone in zones:
        df_train = df.copy()
        df_train.loc[df_train[zone_col] == zone, "in_modeling_sample"] = False
        try:
            df_pred, _ = run_ridge(cfg, df_train)
        except Exception as e:
            logger.warning("LOZO zone %s failed: %s", zone, e)
            continue
        held = df_pred[df_pred[zone_col] == zone].copy()
        if "ridge_moderate" not in held.columns or held["moderate_prevalence"].isna().all():
            continue
        truth = held["moderate_prevalence"].dropna().iloc[0]
        pop = held["population"].fillna(0).values
        pred_vals = held["ridge_moderate"].fillna(0).values
        if pop.sum() > 0:
            pred_mean = np.average(pred_vals, weights=pop)
        else:
            pred_mean = pred_vals.mean()
        errors.append(abs(pred_mean - truth))
    return round(float(np.mean(errors)), 3) if errors else None


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(scales: list[float], skip_lozo: bool = False):
    from src.utils.config_loader import load_config, find_project_root
    from src.pipeline.run_pipeline import run_data_pipeline

    cfg_path = str(ROOT / "config/config_nga.yaml")
    cfg = load_config(cfg_path)

    logger.info("Loading modeling table (cached)…")
    df = run_data_pipeline(cfg, force_rerun=False)

    rows = []
    for scale in scales:
        logger.info("── dhs_aux_dhs_scale = %.4f ─────────────────────────────", scale)
        cfg_run = load_config(cfg_path)

        if scale > 0.0:
            # Use stacked DHS aux; disable soft-label blend for comparability
            cfg_run["modeling"]["ridge"]["dhs_aux_dhs_scale"] = float(scale)
            cfg_run["modeling"]["ridge"]["dhs_aux_mics_scale"] = 1.0
            cfg_run["modeling"]["ridge"]["use_dhs_soft_label"] = False
            tag = f"dhs_stack_scale={scale:.4g}"
        else:
            # Baseline: soft-label blend at configured weight; no DHS stack
            cfg_run["modeling"]["ridge"]["dhs_aux_dhs_scale"] = 0.0
            cfg_run["modeling"]["ridge"]["use_dhs_soft_label"] = True
            tag = f"soft_label_w={cfg_run['modeling']['ridge'].get('dhs_soft_label_weight', 0.4)}"

        # Override n_bootstrap for speed during sweep
        cfg_run["modeling"]["uncertainty"]["n_bootstrap"] = 5

        try:
            from src.models.ridge_model import run as run_ridge
            df_pred, _ = run_ridge(cfg_run, df)
        except Exception as e:
            logger.error("Ridge run failed for scale=%.4g: %s", scale, e)
            rows.append({"tag": tag, "dhs_aux_scale": scale,
                         "dhs_spearman": np.nan, "lozo_mae": np.nan, "error": str(e)})
            continue

        ext = _dhs_external_metrics(df_pred)

        lozo_mae_val = None
        if not skip_lozo:
            logger.info("Running LOZO for scale=%.4g…", scale)
            lozo_mae_val = _lozo_mae(cfg_run, df)

        row = {"tag": tag, "dhs_aux_scale": scale, **ext}
        if not skip_lozo:
            row["lozo_mae"] = lozo_mae_val
        rows.append(row)
        logger.info("  Spearman=%.4f  MAE=%.3f pp  LOZO_MAE=%s",
                    ext["dhs_spearman"], ext["dhs_mae"],
                    f"{lozo_mae_val:.3f}" if lozo_mae_val is not None else "skipped")

    results = pd.DataFrame(rows)
    out_path = str(EVAL_DIR / "dhs_aux_stack_sweep.csv")
    results.to_csv(out_path, index=False)
    logger.info("\nSweep complete. Results:\n%s", results.to_string(index=False))
    logger.info("Saved: %s", out_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep DHS aux-stack scale for Ridge.")
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 1.0],
                        help="dhs_aux_dhs_scale values to test (0 = soft-label baseline).")
    parser.add_argument("--skip-lozo", action="store_true",
                        help="Skip LOZO cross-validation (much faster).")
    args = parser.parse_args()
    run_sweep(scales=args.scales, skip_lozo=args.skip_lozo)


if __name__ == "__main__":
    main()
