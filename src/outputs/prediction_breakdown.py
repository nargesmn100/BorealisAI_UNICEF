"""
Per-cell poverty score breakdown (Ridge linear decomposition + optional GBM SHAP).

Writes ``{output_prefix}_prediction_breakdown.csv`` in ``paths.tables_dir`` and
(through ``merge_breakdown_into_pred_table``) enriches ``{prefix}_predictions.*``
and Folium popups: ``ridge_bdg_popup`` HTML, ``ridge_theme__*`` for LGA-mean
aggregation in ``lga_aggregation`` and the comparison map tooltips.
"""

from __future__ import annotations

import html as html_mod
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from src.utils.config_loader import get_available_features

if TYPE_CHECKING:
    from src.models.ridge_model import RidgeDeprivationModel

logger = logging.getLogger(__name__)

# Group features for theme-level sums (keys are stable slugs; values are lists of column names)
FEATURE_THEME: dict[str, list[str]] = {
    "wealth": ["rwi", "population", "log_population"],
    "urban_built": [
        "smod_class",
        "is_urban",
        "ghsl_built_frac",
        "log_ghsl_built",
        "building_density",
        "log_building_density",
    ],
    "access_services": [
        "travel_time_cities",
        "travel_time_50k",
        "log_travel_time_cities",
        "log_travel_time_50k",
        "dist_school_km",
        "dist_health_km",
    ],
    "nightlights": ["nightlights", "log_nightlights"],
    "health_mics": [
        "anc_rate",
        "skilled_delivery_rate",
        "facility_delivery_rate",
        "vacc_card_rate",
        "diarrhea_care_rate",
    ],
    "education_mics": [
        "school_attendance_rate",
        "ever_attended_rate",
        "public_school_rate",
    ],
    "hazards": ["conflict_events", "conflict_fatalities", "rainfall_mm"],
    "dhs_cluster": ["dhs_nearest_dep_index", "dist_km_nearest_dhs_cluster"],
}

MISSING_GROUP = "other"


def _theme_for_feature(name: str) -> str:
    for theme, members in FEATURE_THEME.items():
        if name in members:
            return theme
    return MISSING_GROUP


def feature_mask(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Rows with complete features; exclude Unknown subregion (matches Ridge pred mask)."""
    fm = df[feature_cols].notna().all(axis=1)
    if "subregion" in df.columns:
        fm = fm & (df["subregion"] != "Unknown")
    return fm


def build_ridge_breakdown_frame(
    df: pd.DataFrame,
    model: "RidgeDeprivationModel",
    feature_cols: list[str],
) -> Optional[pd.DataFrame]:
    """
    Linear decomposition of Ridge on the standardized scale: contribution_j = β_j * z_j.
    Sum of per-feature terms plus intercept matches ``model.predict`` for those rows
    (before any post-hoc RWI add-back, which is not part of the fitted linear term).
    """
    if model.pipeline is None:
        return None
    sub = df.loc[feature_mask(df, feature_cols)].copy()
    if len(sub) == 0:
        return None

    X = sub[feature_cols].values.astype(float)
    scaler = model.pipeline.named_steps["scaler"]
    ridge = model.pipeline.named_steps["ridge"]
    mean = scaler.mean_
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    z = (X - mean) / scale
    coefs = np.asarray(ridge.coef_).ravel()
    intercept = float(ridge.intercept_)

    out = pd.DataFrame()
    if "cell_id" in sub.columns:
        out["cell_id"] = sub["cell_id"].values
    if "latitude" in sub.columns:
        out["latitude"] = sub["latitude"].values
    if "longitude" in sub.columns:
        out["longitude"] = sub["longitude"].values
    if "subregion" in sub.columns:
        out["subregion"] = sub["subregion"].values
    if "population" in sub.columns:
        out["population"] = sub["population"].values
    for c in ("moderate_prevalence", "ridge_moderate", "ridge_raw_score"):
        if c in sub.columns:
            out[c] = sub[c].values

    n = len(sub)
    out["ridge_bdg_intercept"] = np.full(n, intercept)
    contribs = np.zeros((n, len(feature_cols)))
    for j, name in enumerate(feature_cols):
        cj = coefs[j] * z[:, j]
        contribs[:, j] = cj
        out[f"ridge_bdg__{name}"] = cj
        out[f"raw__{name}"] = X[:, j]

    theme_sums: dict[str, np.ndarray] = {t: np.zeros(n) for t in set(FEATURE_THEME) | {MISSING_GROUP}}
    for j, name in enumerate(feature_cols):
        th = _theme_for_feature(name)
        theme_sums[th] = theme_sums[th] + contribs[:, j]
    for t, arr in theme_sums.items():
        out[f"ridge_theme__{t}"] = arr

    linear = intercept + contribs.sum(axis=1)
    out["ridge_bdg_linear_pred"] = linear
    out["ridge_bdg_pred_check_abs_err"] = np.abs(
        np.asarray(model.predict(X)) - linear
    )
    out["ridge_bdg_popup"] = out.apply(
        _format_ridge_breakdown_popup_html, axis=1
    )
    return out


def _format_ridge_breakdown_popup_html(row: pd.Series) -> str:
    """
    Compact HTML for Folium: theme sums and top |feature| contributions (Ridge linear part).
    """
    theme_parts: list[tuple[float, str]] = []
    for k in row.index:
        if not str(k).startswith("ridge_theme__"):
            continue
        v = row[k]
        if pd.isna(v):
            continue
        name = str(k).replace("ridge_theme__", "").replace("_", " ")
        theme_parts.append((abs(float(v)), f"{name}: {float(v):+.2f}"))

    theme_parts.sort(key=lambda x: -x[0])
    theme_str = "<br>".join(
        html_mod.escape(t) for _, t in theme_parts[:5]
    )

    feat_parts: list[tuple[float, str]] = []
    for k in row.index:
        ks = str(k)
        if not ks.startswith("ridge_bdg__") or "theme" in ks:
            continue
        v = row[k]
        if pd.isna(v):
            continue
        fname = ks.replace("ridge_bdg__", "").replace("_", " ")
        feat_parts.append((abs(float(v)), f"{fname}: {float(v):+.2f}"))

    feat_parts.sort(key=lambda x: -x[0])
    top_feat = "<br>".join(
        html_mod.escape(t) for _, t in feat_parts[:4]
    )

    if not theme_str and not top_feat:
        return ""

    out = '<hr style="margin:4px 0"><small><b>Ridge (linear) explain</b><br>'
    if theme_str:
        out += f"<b>Themes (sum β·z):</b><br>{theme_str}<br>"
    if top_feat:
        out += f"<b>Top features:</b><br>{top_feat}</small>"
    return out


def merge_breakdown_into_pred_table(
    pred_table: pd.DataFrame,
    brk: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Add popup + theme columns to the map/pred table for UI (join on ``cell_id``)."""
    if brk is None or len(brk) == 0 or "cell_id" not in pred_table.columns:
        return pred_table
    if "cell_id" not in brk.columns:
        return pred_table
    use_cols = ["cell_id"] + [
        c
        for c in brk.columns
        if c == "ridge_bdg_popup"
        or c.startswith("ridge_theme__")
        or c == "ridge_bdg_intercept"
    ]
    use_cols = [c for c in use_cols if c in brk.columns]
    m = brk[use_cols].drop_duplicates(subset=["cell_id"], keep="last")
    out = pred_table.merge(m, on="cell_id", how="left")
    return out


def maybe_export_gbm_shap(
    cfg: dict,
    df: pd.DataFrame,
    gbm_model: Any,
) -> None:
    """
    Optional per-cell SHAP for GBM (TreeExplainer). Can be memory-heavy; capped by
    ``modeling.gbm_shap_max_cells`` (default 2000).
    """
    mdl_cfg = cfg.get("modeling", {})
    if not mdl_cfg.get("export_gbm_shap", False) or gbm_model is None:
        return
    try:
        import shap
    except ImportError:
        logger.info("shap not installed — skipping export_gbm_shap.")
        return
    feature_cols = get_available_features(cfg, df)
    max_n = int(mdl_cfg.get("gbm_shap_max_cells", 2000))
    rs = cfg.get("modeling", {}).get("gbm", {}).get("random_state", 42)
    mask = feature_mask(df, feature_cols)
    sub = df.loc[mask, feature_cols]
    if len(sub) == 0:
        return
    if len(sub) > max_n:
        sub = sub.sample(n=max_n, random_state=rs)
    X = sub.values.astype(float)
    try:
        explainer = shap.TreeExplainer(gbm_model)
        sv = explainer.shap_values(X)
    except Exception as e:
        logger.warning("GBM SHAP export failed: %s", e)
        return
    out = pd.DataFrame(sv, columns=[f"gbm_shap__{c}" for c in feature_cols])
    base = df.loc[sub.index]
    if "cell_id" in base.columns:
        out["cell_id"] = base["cell_id"].values
    tables_dir = cfg["paths"]["tables_dir"]
    os.makedirs(tables_dir, exist_ok=True)
    prefix = cfg.get("country", {}).get("output_prefix", "out")
    path = os.path.join(tables_dir, f"{prefix}_gbm_shap_sample.csv")
    out.to_csv(path, index=False)
    logger.info("GBM SHAP sample saved (%d rows): %s", len(out), path)


def export_ridge_breakdown(
    cfg: dict,
    df: pd.DataFrame,
    model: "RidgeDeprivationModel",
) -> tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Build Ridge breakdown table and write CSV.
    Returns (path, dataframe) for merging popups into map outputs.
    """
    if not cfg.get("modeling", {}).get("export_prediction_breakdown", True):
        return None, None
    if model is None or model.pipeline is None:
        return None, None
    feature_cols = get_available_features(cfg, df)
    brk = build_ridge_breakdown_frame(df, model, feature_cols)
    if brk is None or len(brk) == 0:
        logger.warning("Ridge prediction breakdown: no rows to export.")
        return None, None
    tables_dir = cfg["paths"]["tables_dir"]
    os.makedirs(tables_dir, exist_ok=True)
    prefix = cfg.get("country", {}).get("output_prefix", "out")
    path = os.path.join(tables_dir, f"{prefix}_prediction_breakdown.csv")
    brk.to_csv(path, index=False)
    err = brk["ridge_bdg_pred_check_abs_err"].max() if "ridge_bdg_pred_check_abs_err" in brk.columns else 0.0
    logger.info(
        "Ridge prediction breakdown saved: %s (%d rows, max |pred−linear|=%.2e).",
        path, len(brk), err,
    )
    if err > 1e-3:
        logger.warning(
            "Ridge linear decomposition check exceed tolerance — inspect scaling/intercept."
        )
    return path, brk
