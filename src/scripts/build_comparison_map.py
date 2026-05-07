"""
Build an interactive multi-panel comparison map for Nigeria child deprivation.

Panels:
  1. MICS 2021 State Truth          — what we train on (state resolution)
  2. Ridge Model Prediction         — ML spatial disaggregation (LGA resolution)
  3. GAM Model Prediction           — additive model (optional, if run)
  4. State-level Aggregate Error    — pop-weighted LGA rollup vs MICS truth
  5. Uncertainty Band               — ridge_upper − ridge_lower
  6. NBS Monetary Poverty           — independent validation (NLSS 2019)
  7. (optional) Absolute Error      — |prediction − truth| at LGA
  8–14. Feature layers (one per group) — see FEATURE_LAYERS list below

All panels shown at LGA level (775 LGAs) with hover tooltips.
An info sidebar explains features, targets, and training/eval split.

Output
------
Data/outputs/nga/maps/nga_comparison_map.html
"""

import html
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = ROOT / "Data/outputs/nga/maps"
MAP_DIR.mkdir(parents=True, exist_ok=True)

try:
    import folium
    from folium import plugins
    from branca.colormap import LinearColormap
except ImportError:
    raise SystemExit("pip install folium branca")


# ---------------------------------------------------------------------------
# Feature layer definitions  (column → display label, colormap colours, fmt)
# ---------------------------------------------------------------------------
# Each entry: (col_in_gdf, layer_label, colours, caption, pct_multiplier, fmt)
#   pct_multiplier: multiply stored value to get display units (e.g. 100 for 0–1 fraction)
#   fmt: Python format string for tooltip value
FEATURE_LAYERS = [
    # ── Wealth & Access ──────────────────────────────────────────────────
    (
        "feat_rwi", "⊕ Feature: RWI (Relative Wealth Index)",
        ["#2166ac", "#abd9e9", "#ffffbf", "#fdae61", "#d73027"],
        "RWI — Relative Wealth Index (higher = wealthier)",
        1.0, "{:.2f}",
        "RWI (pop-wtd LGA mean)", "Meta/World Bank", "~2.4 km grid point",
    ),
    (
        "feat_nightlights", "⊕ Feature: Nightlights (VIIRS 2019)",
        ["#000033", "#0d0d5c", "#3d3db0", "#9999e0", "#ffffd4"],
        "Nightlights — VIIRS radiance (higher = brighter)",
        1.0, "{:.2f}",
        "Nightlights (pop-wtd LGA mean)", "VIIRS 2019", "500 m raster → grid",
    ),
    (
        "feat_travel_time", "⊕ Feature: Travel Time to City (min)",
        ["#006837", "#78c679", "#ffffcc", "#fd8d3c", "#800026"],
        "Travel Time to nearest city ≥ 50 k pop (minutes)",
        1.0, "{:.0f} min",
        "Travel time (pop-wtd LGA mean)", "Oxford accessibility", "1 km raster → grid",
    ),
    # ── Built environment ─────────────────────────────────────────────────
    (
        "feat_ghsl_built", "⊕ Feature: GHSL Built-up Fraction",
        ["#f7f7f7", "#cccccc", "#969696", "#636363", "#252525"],
        "GHSL Built-up Fraction (0–1; higher = more urbanised)",
        100.0, "{:.1f}%",
        "GHSL built-up % (pop-wtd LGA mean)", "GHSL 2020", "100 m raster → grid",
    ),
    (
        "feat_building_density", "⊕ Feature: Building Density",
        ["#f7fcfd", "#ccece6", "#66c2a4", "#238b45", "#00441b"],
        "Building density (buildings per km²)",
        1.0, "{:.1f}",
        "Building density (pop-wtd LGA mean)", "Microsoft/OSM buildings", "Point count → grid",
    ),
    # ── Services & Distance ───────────────────────────────────────────────
    (
        "feat_dist_school", "⊕ Feature: Distance to School (km)",
        ["#006837", "#78c679", "#ffffcc", "#fd8d3c", "#800026"],
        "Distance to nearest school (km; higher = worse access)",
        1.0, "{:.1f} km",
        "Dist. to school (pop-wtd LGA mean)", "OpenStreetMap", "Per grid cell",
    ),
    (
        "feat_dist_health", "⊕ Feature: Distance to Health Facility (km)",
        ["#006837", "#78c679", "#ffffcc", "#fd8d3c", "#800026"],
        "Distance to nearest health facility (km)",
        1.0, "{:.1f} km",
        "Dist. to health (pop-wtd LGA mean)", "OSM + GRID3", "Per grid cell",
    ),
    # ── Conflict ─────────────────────────────────────────────────────────
    (
        "feat_conflict", "⊕ Feature: Conflict Events (ACLED)",
        ["#f7f7f7", "#fdd49e", "#fdbb84", "#fc8d59", "#b30000"],
        "Conflict events (count per cell; higher = more conflict)",
        1.0, "{:.1f}",
        "Conflict events (pop-wtd LGA mean)", "ACLED", "Event points → count per cell",
    ),
    # ── DHS proximity ─────────────────────────────────────────────────────
    (
        "feat_dhs_dep", "⊕ Feature: DHS Deprivation Proximity Index",
        ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
        "DHS nearest cluster deprivation index (higher = more deprived nearby)",
        1.0, "{:.3f}",
        "DHS dep. proximity (pop-wtd LGA mean)", "DHS 2018 GPS clusters", "~1,400 cluster points",
    ),
    # ── State-level admin ─────────────────────────────────────────────────
    (
        "feat_nbs_floor_earth", "⊕ Feature: NBS Floor Earth % (state-level)",
        ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
        "NBS: % HH with earth floor (state-level constant)",
        100.0, "{:.1f}%",
        "NBS floor earth % (state constant)", "NBS MPI survey", "State-level",
    ),
    (
        "feat_nemis_primary_pps", "⊕ Feature: NEMIS Primary Pupils/School (state-level)",
        ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
        "NEMIS: primary school pupils per school (state-level)",
        1.0, "{:.0f}",
        "NEMIS pupils/school (state constant)", "NEMIS", "State-level",
    ),
]


# ---------------------------------------------------------------------------
# Cell-level feature columns to load from the consolidated parquet
# ---------------------------------------------------------------------------
_CELL_FEAT_COLS = [
    "latitude", "longitude", "population", "parish_name",
    "rwi", "nightlights", "travel_time_50k",
    "ghsl_built_frac", "building_density",
    "dist_school_km", "dist_health_km",
    "conflict_events",
    "dhs_nearest_dep_index",
    "nbs_floor_earth_pct",
    "nemis_primary_pupil_per_school",
]

# Map: feat_<col> in gdf  →  raw column in parquet
_FEAT_COL_MAP = {
    "feat_rwi":               "rwi",
    "feat_nightlights":       "nightlights",
    "feat_travel_time":       "travel_time_50k",
    "feat_ghsl_built":        "ghsl_built_frac",
    "feat_building_density":  "building_density",
    "feat_dist_school":       "dist_school_km",
    "feat_dist_health":       "dist_health_km",
    "feat_conflict":          "conflict_events",
    "feat_dhs_dep":           "dhs_nearest_dep_index",
    "feat_nbs_floor_earth":   "nbs_floor_earth_pct",
    "feat_nemis_primary_pps": "nemis_primary_pupil_per_school",
}

# State-level features (same value for every cell in a state → show as state polygon)
_STATE_LEVEL_FEATS = {"nbs_floor_earth_pct", "nemis_primary_pupil_per_school"}

# Raw-resolution feature layers to show as cell dots (subset for performance)
# (col_in_parquet, layer_label, colours, caption, fmt, tip_label, source, resolution)
RAW_CELL_LAYERS = [
    (
        "rwi",
        "◉ Raw: RWI Wealth Index (~2.4 km grid)",
        ["#2166ac", "#abd9e9", "#ffffbf", "#fdae61", "#d73027"],
        "RWI — Relative Wealth Index at native ~2.4 km grid",
        "{:.2f}", "RWI", "Meta / World Bank RWI", "~2.4 km grid point",
    ),
    (
        "nightlights",
        "◉ Raw: Nightlights — VIIRS 2019 (500 m → grid)",
        ["#000033", "#1a1a6e", "#4040b0", "#9999e0", "#ffffd4"],
        "VIIRS Nightlights radiance (at RWI grid; source: 500 m raster)",
        "{:.2f}", "Nightlights", "VIIRS 2019", "500 m raster sampled at ~2.4 km grid",
    ),
    (
        "travel_time_50k",
        "◉ Raw: Travel Time to City (1 km → grid)",
        ["#1a9850", "#91cf60", "#ffffbf", "#fc8d59", "#d73027"],
        "Travel time (min) to city ≥ 50 k pop (source: 1 km raster)",
        "{:.0f} min", "Travel time", "Oxford accessibility", "1 km raster sampled at ~2.4 km grid",
    ),
    (
        "ghsl_built_frac",
        "◉ Raw: GHSL Built-up Fraction (100 m → grid)",
        ["#f7f7f7", "#cccccc", "#969696", "#525252", "#252525"],
        "GHSL built-up fraction 0–1 (source: 100 m raster)",
        "{:.3f}", "Built-up fraction", "GHSL 2020", "100 m raster sampled at ~2.4 km grid",
    ),
    (
        "dist_school_km",
        "◉ Raw: Distance to School (per cell, km)",
        ["#1a9850", "#91cf60", "#ffffbf", "#fc8d59", "#d73027"],
        "Distance to nearest school (km) per RWI grid cell",
        "{:.1f} km", "Dist. to school", "OpenStreetMap schools", "Computed per ~2.4 km grid cell",
    ),
    (
        "dhs_nearest_dep_index",
        "◉ Raw: DHS Deprivation Proximity (per cell)",
        ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
        "Deprivation index of nearest DHS cluster (higher = more deprived nearby)",
        "{:.3f}", "DHS dep. proximity", "DHS 2018 GPS clusters (~1,400)", "Per ~2.4 km grid cell",
    ),
]

# State-admin raw layers  (col_in_parquet, label, colours, caption, fmt, tip_label)
RAW_STATE_LAYERS = [
    (
        "nbs_floor_earth_pct",
        "◉ Raw: NBS Floor Earth % (state polygon)",
        ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
        "NBS: % HH with earth floor — state-level constant (NBS MPI survey)",
        "{:.1%}", "NBS floor earth %",
        "NBS MPI survey (~53k HH)", "State-level (37 states)",
    ),
    (
        "nemis_primary_pupil_per_school",
        "◉ Raw: NEMIS Pupils/School (state polygon)",
        ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
        "NEMIS: primary school pupils per school — state-level constant",
        "{:.0f}", "NEMIS pupils/school",
        "NEMIS (~180k schools)", "State-level (37 states)",
    ),
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    gdf = gpd.read_file(ROOT / "Data/outputs/nga/maps/nga_lga_predictions.geojson")

    # Merge NBS state poverty
    nbs_path = ROOT / "Data/Nigeria/nbs/nga_nbs_state_poverty.csv"
    if nbs_path.exists():
        nbs = pd.read_csv(nbs_path)[["state", "poverty_headcount_pct"]]
        nbs["state"] = nbs["state"].replace({
            "Federal Capital Territory": "FCT Abuja",
            "Cross River": "Cross River",
        })
        gdf = gdf.merge(nbs, on="state", how="left")
    else:
        gdf["poverty_headcount_pct"] = np.nan

    # LGA-level error (prediction vs MICS truth propagated to each LGA in state)
    gdf["ridge_abs_error"] = (gdf["ridge_moderate"] - gdf["mics_state_truth"]).abs()
    gdf["uncertainty_band"] = gdf["ridge_moderate_upper"] - gdf["ridge_moderate_lower"]

    # ── State-level aggregate error ────────────────────────────────────────
    def _state_agg(g):
        valid = g[g["ridge_moderate"].notna() & (g["ridge_moderate"] > 0)]
        if valid["total_population"].sum() > 0:
            pred = np.average(valid["ridge_moderate"], weights=valid["total_population"])
        else:
            pred = 0.0
        truth = g["mics_state_truth"].iloc[0]
        coverage = len(valid) / len(g) * 100
        return pd.Series({
            "state_ridge_agg": pred,
            "state_truth":     truth,
            "state_agg_abs_error": abs(pred - truth) if pd.notna(truth) else np.nan,
            "state_agg_signed_error": pred - truth if pd.notna(truth) else np.nan,
            "state_lga_coverage_pct": coverage,
        })

    state_err = (
        gdf.groupby("state")
        .apply(_state_agg, include_groups=False)
        .reset_index()
    )
    gdf = gdf.merge(state_err, on="state", how="left")

    # ── Load cell data once; use for both LGA aggregation + raw dot layers ──
    cells_df = _load_cell_data()
    if cells_df is not None:
        gdf = _merge_feature_columns(gdf, cells_df)
        cells_sample = _stratified_sample(cells_df, n_per_state=250)
        state_df = _state_level_features(cells_df)
    else:
        _nullify_feature_cols(gdf)
        cells_sample = None
        state_df = None

    return gdf, cells_sample, state_df


def _load_cell_data() -> pd.DataFrame | None:
    parquet_path = ROOT / "Data/outputs/nga/tables/nga_full_consolidated.parquet"
    if not parquet_path.exists():
        print("  [warn] Consolidated parquet not found — skipping feature layers.")
        return None
    print("  Loading consolidated parquet …")
    df = pd.read_parquet(parquet_path, columns=_CELL_FEAT_COLS)
    df = df.dropna(subset=["latitude", "longitude"])
    return df


def _stratified_sample(df: pd.DataFrame, n_per_state: int = 250) -> pd.DataFrame:
    """Return a stratified sample (~n_per_state cells per state) for dot layers."""
    parts = []
    for _, grp in df.groupby("parish_name"):
        n = min(len(grp), n_per_state)
        parts.append(grp.sample(n=n, random_state=42))
    sampled = pd.concat(parts, ignore_index=True)
    print(f"  Stratified sample: {len(sampled):,} cells from {df['parish_name'].nunique()} states")
    return sampled


def _state_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per state with state-level feature constants."""
    return (
        df.groupby("parish_name")[
            ["nbs_floor_earth_pct", "nemis_primary_pupil_per_school"]
        ]
        .first()
        .reset_index()
        .rename(columns={"parish_name": "state"})
    )


def _merge_feature_columns(
    gdf: gpd.GeoDataFrame, cells_df: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Spatial-join cells to LGAs; pop-weight aggregate each feature."""
    lga_ref = gdf[["lga_id", "geometry"]].copy()
    cell_gdf = gpd.GeoDataFrame(
        cells_df,
        geometry=gpd.points_from_xy(cells_df["longitude"], cells_df["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(cell_gdf, lga_ref, how="left", predicate="within")
    joined = joined.dropna(subset=["lga_id"])

    rows = []
    for lga_id_val, grp in joined.groupby("lga_id"):
        pop = grp["population"].fillna(0).clip(lower=0).values
        pop_sum = pop.sum()
        row = {"lga_id": lga_id_val}
        for feat_col, src_col in _FEAT_COL_MAP.items():
            if src_col not in grp.columns:
                row[feat_col] = np.nan
                continue
            vals = grp[src_col].values.astype(float)
            ok = np.isfinite(vals) & (pop > 0)
            if ok.any() and pop[ok].sum() > 0:
                row[feat_col] = float(np.average(vals[ok], weights=pop[ok]))
            else:
                row[feat_col] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    gdf = gdf.merge(feat_df, on="lga_id", how="left")
    print(f"  Feature columns merged for {len(feat_df)} LGAs.")
    return gdf


def _nullify_feature_cols(gdf: gpd.GeoDataFrame) -> None:
    for col, *_ in FEATURE_LAYERS:
        gdf[col] = np.nan


# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------
DEPRIV_COLOURS = ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"]
ERROR_COLOURS  = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]
UNC_COLOURS    = ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"]


def make_colormap(vmin, vmax, colors, caption):
    return LinearColormap(colors, vmin=vmin, vmax=vmax, caption=caption)


def lga_ridge_theme_tooltip_line(row) -> str:
    """One line of pop-weighted mean Ridge theme contributions (LGA)."""
    parts: list[tuple[float, str]] = []
    for k in row.index:
        if not str(k).startswith("ridge_theme__"):
            continue
        v = row[k]
        if pd.isna(v):
            continue
        name = str(k).replace("ridge_theme__", "").replace("_", " ")
        parts.append((abs(float(v)), f"{name}: {float(v):+.2f}"))
    if not parts:
        return ""
    parts.sort(key=lambda x: -x[0])
    line = " &nbsp;|&nbsp; ".join(t for _, t in parts[:4])
    return f"<br><small><b>Ridge themes (LGA):</b> {html.escape(line)}</small>"


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------
def add_choropleth_layer(m, gdf, value_col, colormap, layer_name, tooltip_cols):
    """Add a GeoJSON choropleth layer with hover tooltips."""
    fg = folium.FeatureGroup(name=layer_name, show=False)

    geojson_data = json.loads(gdf.to_json())

    for feature, (_, row) in zip(geojson_data["features"], gdf.iterrows()):
        val = row[value_col]
        color = colormap(val) if pd.notna(val) else "#cccccc"

        tooltip_html = "<br>".join(
            f"<b>{k}:</b> {row[k]:.1f}%" if isinstance(row[k], float) and k.endswith("_pct") or "moderate" in k or "truth" in k or "error" in k or "band" in k
            else f"<b>{k}:</b> {row[k]}"
            for k in tooltip_cols if k in row.index
        )

        folium.GeoJson(
            feature,
            style_function=lambda f, c=color: {
                "fillColor": c,
                "color": "#555555",
                "weight": 0.4,
                "fillOpacity": 0.75,
            },
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
        ).add_to(fg)

    fg.add_to(m)
    colormap.add_to(m)
    return fg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_map(
    gdf: gpd.GeoDataFrame,
    cells_sample: pd.DataFrame | None = None,
    state_df: pd.DataFrame | None = None,
) -> folium.Map:
    center = [9.5, 8.0]
    m = folium.Map(
        location=center,
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # ── Common tooltip fields ──────────────────────────────────────────────
    base_tooltip = ["state", "lga_name", "mics_state_truth",
                    "ridge_moderate", "gam_moderate", "ridge_abs_error",
                    "uncertainty_band", "total_population"]

    # ── Panel 1: MICS Truth ───────────────────────────────────────────────
    vmin, vmax = 10, 75
    cm1 = make_colormap(vmin, vmax, DEPRIV_COLOURS, "MICS Truth — Moderate Deprivation (%)")
    fg1 = folium.FeatureGroup(name="① MICS 2021 State Truth", show=True)
    for _, row in gdf.iterrows():
        val = row["mics_state_truth"]
        color = cm1(np.clip(val, vmin, vmax)) if pd.notna(val) else "#cccccc"
        tip = (
            f"<b>State:</b> {row['state']}<br>"
            f"<b>LGA:</b> {row['lga_name']}<br>"
            f"<b>MICS Truth:</b> {val:.1f}%<br>"
            f"<b>Population:</b> {row['total_population']:,.0f}"
        )
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=color: {"fillColor": c, "color": "#444", "weight": 0.4, "fillOpacity": 0.78},
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(fg1)
    fg1.add_to(m)
    cm1.add_to(m)

    # ── Panel 2: Ridge Prediction ──────────────────────────────────────────
    cm2 = make_colormap(vmin, vmax, DEPRIV_COLOURS, "Ridge Prediction — Moderate Deprivation (%)")
    fg2 = folium.FeatureGroup(name="② Ridge Model Prediction", show=False)
    for _, row in gdf.iterrows():
        val = row["ridge_moderate"]
        color = cm2(np.clip(val, vmin, vmax)) if pd.notna(val) else "#cccccc"
        tip = (
            f"<b>State:</b> {row['state']}<br>"
            f"<b>LGA:</b> {row['lga_name']}<br>"
            f"<b>Ridge:</b> {val:.1f}%<br>"
            f"<b>MICS Truth:</b> {row['mics_state_truth']:.1f}%<br>"
            f"<b>Error:</b> {row['ridge_abs_error']:.1f} pp"
        )
        tip += lga_ridge_theme_tooltip_line(row)
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=color: {"fillColor": c, "color": "#444", "weight": 0.4, "fillOpacity": 0.78},
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(fg2)
    fg2.add_to(m)
    cm2.add_to(m)

    # ── Panel 3 (optional): GAM Prediction ─────────────────────────────────
    # Some runs skip GAM entirely; only render this panel if the column exists.
    if "gam_moderate" in gdf.columns and gdf["gam_moderate"].notna().any():
        cm3 = make_colormap(vmin, vmax, DEPRIV_COLOURS, "GAM Prediction — Moderate Deprivation (%)")
        fg3 = folium.FeatureGroup(name="③ GAM Model Prediction", show=False)
        for _, row in gdf.iterrows():
            val = row["gam_moderate"]
            color = cm3(np.clip(val, vmin, vmax)) if pd.notna(val) else "#cccccc"
            tip = (
                f"<b>State:</b> {row['state']}<br>"
                f"<b>LGA:</b> {row['lga_name']}<br>"
                f"<b>GAM:</b> {val:.1f}%<br>"
                f"<b>MICS Truth:</b> {row['mics_state_truth']:.1f}%"
            )
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, c=color: {"fillColor": c, "color": "#444", "weight": 0.4, "fillOpacity": 0.78},
                tooltip=folium.Tooltip(tip, sticky=True),
            ).add_to(fg3)
        fg3.add_to(m)
        cm3.add_to(m)

    # ── Panel 4: State-level aggregate error ──────────────────────────────
    # Shows: pop-weight all LGA ridge predictions to state → compare to MICS truth.
    # Reconciled states will be near 0. States with missing LGA data show large gaps.
    err_max2 = float(gdf["state_agg_abs_error"].dropna().quantile(0.95))
    cm4b = make_colormap(0, max(err_max2, 1), ERROR_COLOURS,
                         "State-Aggregate Error (pp) — Σ(LGA preds) vs MICS truth")
    fg4b = folium.FeatureGroup(name="④ State Aggregate Error (LGA→State vs MICS)", show=False)
    for _, row in gdf.iterrows():
        val = row["state_agg_abs_error"]
        color = cm4b(np.clip(val, 0, err_max2)) if pd.notna(val) else "#cccccc"
        cov = row.get("state_lga_coverage_pct", np.nan)
        cov_s = f"{cov:.0f}%" if pd.notna(cov) else "N/A"
        pred_s = f"{row['state_ridge_agg']:.1f}%" if pd.notna(row['state_ridge_agg']) else "N/A"
        truth_s = f"{row['state_truth']:.1f}%" if pd.notna(row['state_truth']) else "N/A"
        signed = row.get("state_agg_signed_error", np.nan)
        signed_s = f"{signed:+.2f} pp" if pd.notna(signed) else "N/A"
        tip = (
            f"<b>State:</b> {row['state']}<br>"
            f"<b>LGA:</b> {row['lga_name']}<br>"
            f"<b>MICS truth (state):</b> {truth_s}<br>"
            f"<b>Pop-wtd LGA aggregate:</b> {pred_s}<br>"
            f"<b>Absolute error:</b> {val:.2f} pp<br>"
            f"<b>Signed error:</b> {signed_s}<br>"
            f"<b>LGA coverage:</b> {cov_s}<br>"
            f"<small style='color:#888'>Near 0 = reconciliation working;<br>"
            f"Large = LGAs in this state had missing predictions.</small>"
        )
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=color: {"fillColor": c, "color": "#444", "weight": 0.4, "fillOpacity": 0.78},
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(fg4b)
    fg4b.add_to(m)
    cm4b.add_to(m)

    # ── Panel 5: Absolute Error (LGA level, original) ─────────────────────
    err_max = float(gdf["ridge_abs_error"].quantile(0.95))
    cm4 = make_colormap(0, max(err_max, 5), ERROR_COLOURS, "Ridge Absolute Error (pp) — LGA level")
    fg4 = folium.FeatureGroup(name="⑤ Ridge Abs Error (LGA-level, vs propagated state truth)", show=False)
    for _, row in gdf.iterrows():
        val = row["ridge_abs_error"]
        color = cm4(np.clip(val, 0, err_max)) if pd.notna(val) else "#cccccc"
        tip = (
            f"<b>State:</b> {row['state']}<br>"
            f"<b>LGA:</b> {row['lga_name']}<br>"
            f"<b>|Error|:</b> {val:.1f} pp<br>"
            f"<b>LGA Predicted:</b> {row['ridge_moderate']:.1f}%<br>"
            f"<b>State Truth:</b> {row['mics_state_truth']:.1f}%<br>"
            f"<small style='color:#888'>Note: truth is at state level;<br>"
            f"LGA errors reflect within-state spatial spread.</small>"
        )
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=color: {"fillColor": c, "color": "#444", "weight": 0.4, "fillOpacity": 0.78},
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(fg4)
    fg4.add_to(m)
    cm4.add_to(m)

    # ── Panel 6: Uncertainty ───────────────────────────────────────────────
    unc_max = float(gdf["uncertainty_band"].quantile(0.95))
    cm5 = make_colormap(0, max(unc_max, 1), UNC_COLOURS, "Prediction Uncertainty (CI width, pp)")
    fg5 = folium.FeatureGroup(name="⑥ Uncertainty Band (CI width)", show=False)
    for _, row in gdf.iterrows():
        val = row["uncertainty_band"]
        color = cm5(np.clip(val, 0, unc_max)) if pd.notna(val) else "#cccccc"
        tip = (
            f"<b>State:</b> {row['state']}<br>"
            f"<b>LGA:</b> {row['lga_name']}<br>"
            f"<b>CI width:</b> {val:.2f} pp<br>"
            f"<b>90% CI:</b> [{row['ridge_moderate_lower']:.1f}, {row['ridge_moderate_upper']:.1f}]"
        )
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=color: {"fillColor": c, "color": "#444", "weight": 0.4, "fillOpacity": 0.78},
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(fg5)
    fg5.add_to(m)
    cm5.add_to(m)

    # ── Panel 7: NBS Monetary Poverty ─────────────────────────────────────
    if "poverty_headcount_pct" in gdf.columns and gdf["poverty_headcount_pct"].notna().any():
        nbs_max = float(gdf["poverty_headcount_pct"].max())
        cm6 = make_colormap(0, nbs_max, DEPRIV_COLOURS, "NBS NLSS 2019 — Monetary Poverty (%)")
        fg6 = folium.FeatureGroup(name="⑦ NBS NLSS 2019 Monetary Poverty", show=False)
        for _, row in gdf.iterrows():
            val = row["poverty_headcount_pct"]
            color = cm6(np.clip(val, 0, nbs_max)) if pd.notna(val) else "#cccccc"
            tip = (
                f"<b>State:</b> {row['state']}<br>"
                f"<b>LGA:</b> {row['lga_name']}<br>"
                f"<b>NBS Monetary Poverty:</b> {val:.1f}%<br>"
                f"<b>MICS Multidimensional:</b> {row['mics_state_truth']:.1f}%"
            )
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, c=color: {"fillColor": c, "color": "#444", "weight": 0.4, "fillOpacity": 0.78},
                tooltip=folium.Tooltip(tip, sticky=True),
            ).add_to(fg6)
        fg6.add_to(m)
        cm6.add_to(m)

    # ── Feature input layers — LGA-aggregated (pop-weighted means) ───────
    for (col, layer_label, colours, caption, pct_mult, fmt,
         tip_label, source, resolution) in FEATURE_LAYERS:
        if col not in gdf.columns or gdf[col].isna().all():
            continue
        vals = gdf[col].dropna()
        if len(vals) == 0:
            continue
        vmin_f = float(vals.quantile(0.02))
        vmax_f = float(vals.quantile(0.98))
        if vmax_f <= vmin_f:
            vmax_f = vmin_f + 1.0
        cm_f = make_colormap(vmin_f, vmax_f, colours, caption)
        fg_f = folium.FeatureGroup(name=layer_label, show=False)
        for _, row in gdf.iterrows():
            raw_val = row[col]
            color = cm_f(float(np.clip(raw_val, vmin_f, vmax_f))) if pd.notna(raw_val) else "#cccccc"
            disp_val = raw_val * pct_mult if pd.notna(raw_val) else None
            val_str  = fmt.format(disp_val) if disp_val is not None else "N/A"
            tip = (
                f"<b>State:</b> {row['state']}<br>"
                f"<b>LGA:</b> {row['lga_name']}<br>"
                f"<b>{tip_label} (LGA pop-wtd avg):</b> {val_str}<br>"
                f"<small style='color:#999'>Aggregated from ~2.4 km cell grid<br>"
                f"Source: {source} | Native res: {resolution}</small>"
            )
            folium.GeoJson(
                row["geometry"].__geo_interface__,
                style_function=lambda f, c=color: {
                    "fillColor": c, "color": "#444", "weight": 0.4, "fillOpacity": 0.80,
                },
                tooltip=folium.Tooltip(tip, sticky=True),
            ).add_to(fg_f)
        fg_f.add_to(m)
        cm_f.add_to(m)

    # ── Feature input layers — RAW cell dots at native ~2.4 km resolution ─
    if cells_sample is not None and len(cells_sample) > 0:
        for (src_col, layer_label, colours, caption, fmt,
             tip_label, source, resolution) in RAW_CELL_LAYERS:
            if src_col not in cells_sample.columns:
                continue
            vals_raw = cells_sample[src_col].dropna()
            if len(vals_raw) == 0:
                continue
            vmin_r = float(vals_raw.quantile(0.02))
            vmax_r = float(vals_raw.quantile(0.98))
            if vmax_r <= vmin_r:
                vmax_r = vmin_r + 1.0
            cm_r = make_colormap(vmin_r, vmax_r, colours, caption)
            fg_r = folium.FeatureGroup(name=layer_label, show=False)
            for _, row in cells_sample.iterrows():
                raw_val = row[src_col]
                if pd.isna(raw_val):
                    continue
                color = cm_r(float(np.clip(raw_val, vmin_r, vmax_r)))
                val_str = fmt.format(raw_val)
                pop_val = row["population"] if pd.notna(row["population"]) else 0
                tip = (
                    f"<b>{tip_label}:</b> {val_str}<br>"
                    f"<b>Lat/Lon:</b> {row['latitude']:.4f}, {row['longitude']:.4f}<br>"
                    f"<b>Population:</b> {pop_val:,.0f}<br>"
                    f"<small style='color:#666'>"
                    f"<b>This is the native resolution</b><br>"
                    f"Source: {source}<br>"
                    f"Grid spacing: ~2.4 km (RWI lattice)</small>"
                )
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    weight=0,
                    tooltip=folium.Tooltip(tip, sticky=True),
                ).add_to(fg_r)
            fg_r.add_to(m)
            cm_r.add_to(m)

    # ── Feature input layers — state-level admin (NBS / NEMIS) ───────────
    # State-admin features are a single value per state — render as state polygon
    if state_df is not None and len(state_df) > 0:
        state_gdf = gdf[["state", "lga_name", "geometry", "total_population"]].copy()
        state_dissolved = (
            state_gdf.dissolve(by="state", aggfunc={"total_population": "sum"})
            .reset_index()
        )
        state_dissolved = state_dissolved.merge(state_df, on="state", how="left")

        for (src_col, layer_label, colours, caption, fmt,
             tip_label, source, resolution) in RAW_STATE_LAYERS:
            if src_col not in state_dissolved.columns:
                continue
            vals_s = state_dissolved[src_col].dropna()
            if len(vals_s) == 0:
                continue
            vmin_s = float(vals_s.min())
            vmax_s = float(vals_s.max())
            if vmax_s <= vmin_s:
                vmax_s = vmin_s + 1.0
            cm_s = make_colormap(vmin_s, vmax_s, colours, caption)
            fg_s = folium.FeatureGroup(name=layer_label, show=False)
            for _, row in state_dissolved.iterrows():
                raw_val = row[src_col]
                color = cm_s(float(np.clip(raw_val, vmin_s, vmax_s))) if pd.notna(raw_val) else "#cccccc"
                val_str = fmt.format(raw_val) if pd.notna(raw_val) else "N/A"
                tip = (
                    f"<b>State:</b> {row['state']}<br>"
                    f"<b>{tip_label}:</b> {val_str}<br>"
                    f"<small style='color:#666'>"
                    f"<b>State-level constant</b> — all LGAs in this state share this value<br>"
                    f"Source: {source}<br>"
                    f"Resolution: {resolution}</small>"
                )
                folium.GeoJson(
                    row["geometry"].__geo_interface__,
                    style_function=lambda f, c=color: {
                        "fillColor": c, "color": "#555", "weight": 1.0, "fillOpacity": 0.82,
                    },
                    tooltip=folium.Tooltip(tip, sticky=True),
                ).add_to(fg_s)
            fg_s.add_to(m)
            cm_s.add_to(m)

    # ── Layer control ──────────────────────────────────────────────────────
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    # ── Legend toggle button ───────────────────────────────────────────────
    # branca colourmap bars land in <div class="legend"> elements; this button
    # hides/shows all of them so they don't clutter the view.
    legend_toggle_html = """
    <style>
      #legend-toggle-btn {
        position: fixed;
        bottom: 28px;
        right: 12px;
        z-index: 2000;
        background: #fff;
        border: 1px solid #aaa;
        border-radius: 5px;
        padding: 5px 11px;
        font-size: 12px;
        font-family: Arial, sans-serif;
        cursor: pointer;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.25);
        user-select: none;
      }
      #legend-toggle-btn:hover { background: #f0f0f0; }
    </style>
    <button id="legend-toggle-btn" onclick="toggleLegends(this)">
      🎨 Hide legends
    </button>
    <script>
      function toggleLegends(btn) {
        // branca renders colormaps as elements with class "legend"
        var legends = document.querySelectorAll('.legend');
        var hidden = btn.dataset.hidden === '1';
        legends.forEach(function(el) {
          el.style.display = hidden ? '' : 'none';
        });
        btn.dataset.hidden = hidden ? '0' : '1';
        btn.textContent = hidden ? '🎨 Hide legends' : '🎨 Show legends';
      }
    </script>
    """
    m.get_root().html.add_child(folium.Element(legend_toggle_html))

    # ── Info sidebar ───────────────────────────────────────────────────────
    info_html = """
    <div id="info-sidebar" style="
        position: fixed; top: 12px; left: 60px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif; max-width: 340px;
        max-height: 92vh; overflow-y: auto; font-size: 11px;">

      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <b style="font-size:13px;">Nigeria Child Deprivation — LGA Level</b>
        <button onclick="document.getElementById('info-sidebar').style.display='none'"
                style="border:none;background:#eee;border-radius:4px;padding:2px 7px;cursor:pointer;font-size:11px;">✕</button>
      </div>

      <!-- LAYERS -->
      <div style="background:#f5f5f5;border-radius:4px;padding:6px 8px;margin-bottom:8px;">
        <b>Layers (top-right ▶ to toggle)</b><br>
        <b style="color:#555">① – ⑦ Model outputs (LGA polygons)</b><br>
        ① MICS 2021 state truth &nbsp; ② Ridge prediction<br>
        ④ State aggregate error &nbsp; ⑤ LGA error &nbsp; ⑥ CI width<br>
        <b style="color:#555;margin-top:4px;display:block">⊕ Feature inputs — LGA avg (pop-weighted)</b>
        RWI · Nightlights · Travel time<br>
        GHSL built-up · Building density<br>
        Dist. to school · Dist. to health<br>
        Conflict events · DHS dep. proximity<br>
        NBS floor earth · NEMIS pupils/school<br>
        <b style="color:#1a7a1a;margin-top:4px;display:block">◉ Feature inputs — RAW native resolution</b>
        <span style="color:#1a7a1a">◉ RWI · Nightlights · Travel time<br>
        ◉ GHSL built-up · Dist. to school<br>
        ◉ DHS dep. proximity<br>
        ◉ NBS floor earth (state polygon)<br>
        ◉ NEMIS pupils/school (state polygon)</span><br>
        <small style="color:#888">◉ layers = dots at actual ~2.4 km grid spacing</small>
      </div>

      <!-- TARGETS -->
      <details open>
        <summary style="cursor:pointer;font-weight:bold;margin-bottom:4px;">
          Target variables &amp; resolution
        </summary>
        <table style="width:100%;border-collapse:collapse;font-size:10.5px;">
          <tr style="background:#f0f0f0;">
            <th style="text-align:left;padding:3px 4px;border-bottom:1px solid #ddd;">Dimension</th>
            <th style="text-align:left;padding:3px 4px;border-bottom:1px solid #ddd;">Ground truth</th>
            <th style="text-align:left;padding:3px 4px;border-bottom:1px solid #ddd;">Prediction res.</th>
          </tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>Composite (MPI)</b></td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">37 states (MICS6 2021)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">LGA (775) + grid (~2.4 km)</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">Shelter</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">37 states (MICS6)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">LGA + grid</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">Sanitation</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">37 states (MICS6)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">LGA + grid</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">Water</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">37 states (MICS6)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">LGA + grid</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">Nutrition (HAZ)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">37 states (DHS 2018)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">LGA + grid</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">Education 5–14</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">37 states (MICS6)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">LGA + grid</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">Education 15–17</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">37 states (MICS6)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">LGA + grid</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">Health 12–35m</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">37 states (MICS6)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">LGA + grid</td></tr>
          <tr><td style="padding:2px 4px;">Health 36–59m</td>
              <td style="padding:2px 4px;">37 states (MICS6)</td>
              <td style="padding:2px 4px;">LGA + grid</td></tr>
        </table>
        <div style="margin-top:4px;color:#666;font-size:10px;">
          <b>No LGA-level ground truth exists.</b> The model disaggregates state
          totals to LGA/grid using spatial signals. Reconciliation forces the
          population-weighted mean of LGA predictions to match the state target.
        </div>
      </details>

      <!-- TRAINING / EVAL -->
      <details style="margin-top:8px;">
        <summary style="cursor:pointer;font-weight:bold;margin-bottom:4px;">
          Training vs evaluation split
        </summary>
        <table style="width:100%;border-collapse:collapse;font-size:10.5px;">
          <tr style="background:#f0f0f0;">
            <th style="text-align:left;padding:3px 4px;border-bottom:1px solid #ddd;">Split</th>
            <th style="text-align:left;padding:3px 4px;border-bottom:1px solid #ddd;">What</th>
            <th style="text-align:left;padding:3px 4px;border-bottom:1px solid #ddd;">Detail</th>
          </tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>Training</b></td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">All 37 states</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Every cell used to fit Ridge + DHS cluster aux loss. No withheld zone in main model.</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>LOZO CV</b></td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Leave-One-State-Out</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Each state held out in turn; model trained on 36, evaluated raw (no reconciliation). Median AE 9.1 pp.</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>Hierarchical CV</b></td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">6 zones → 37 states</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Trained on 6 zone labels; predicted all 37 states. MAE 7.5 pp, r = 0.85.</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;"><b>External</b></td>
              <td style="padding:2px 4px;">DHS 2018 GPS clusters</td>
              <td style="padding:2px 4px;">~1,400 clusters not used in training. Spearman ρ = 0.60.</td></tr>
        </table>
        <div style="margin-top:4px;color:#666;font-size:10px;">
          <b>Implication:</b> The main model is trained on all 37 states — no geographic
          area is fully withheld from training. LOZO and hierarchical CV quantify
          how well the spatial pattern generalises to unseen states/regions.
        </div>
      </details>

      <!-- FEATURES -->
      <details style="margin-top:8px;">
        <summary style="cursor:pointer;font-weight:bold;margin-bottom:4px;">
          Feature inputs &amp; resolution (38 total)
        </summary>
        <table style="width:100%;border-collapse:collapse;font-size:10.5px;">
          <tr style="background:#f0f0f0;">
            <th style="text-align:left;padding:3px 4px;border-bottom:1px solid #ddd;">Feature</th>
            <th style="text-align:left;padding:3px 4px;border-bottom:1px solid #ddd;">Source</th>
            <th style="text-align:left;padding:3px 4px;border-bottom:1px solid #ddd;">Native res.</th>
          </tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">rwi</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Meta / World Bank RWI</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">~2.4 km grid point</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">population, log_population</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">WorldPop 2020</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">100 m raster → sampled at grid</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">smod_class, is_urban</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">GHSL SMOD 2020</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">1 km raster → sampled at grid</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">travel_time_cities/50k (×log)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Oxford accessibility rasters</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">1 km raster → sampled at grid</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">nightlights, log_nightlights</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">VIIRS 2019</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">500 m raster → sampled at grid</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">building_density, log_building_density</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Microsoft/OSM buildings</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Point count → density at grid</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">ghsl_built_frac, log_ghsl_built</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">GHSL built surface 2020</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">100 m raster → sampled at grid</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">dist_school_km</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">OpenStreetMap schools</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Point distance per grid cell</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">dist_health_km</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">OSM + GRID3 health facilities</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Point distance per grid cell</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">conflict_events, conflict_fatalities</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">ACLED</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Event points → count per cell</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">rainfall_mm</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">TerraClimate 2018</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">~4 km raster → sampled at grid</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">dhs_nearest_dep_index</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">DHS 2018 GPS clusters</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">~1,400 cluster points (jittered ±5 km)</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">dist_km_nearest_dhs_cluster</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">DHS 2018 GPS clusters</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">Distance per grid cell</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">nbs_floor_earth/finished_pct</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">NBS MPI survey (~53k HH)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>State-level</b> (joined to all cells in state)</td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">nbs_water_improved/far_pct</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">NBS MPI survey</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>State-level</b></td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">nbs_toilet_improved/open_defecation_pct</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">NBS MPI survey</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>State-level</b></td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">nbs_food_insecure/severe_pct</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">NBS MPI survey</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>State-level</b></td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;border-bottom:1px solid #eee;">nbs_health_far_pct</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">NBS MPI survey</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>State-level</b></td></tr>
          <tr><td style="padding:2px 4px;border-bottom:1px solid #eee;">nemis_primary_schools/enrol/pupil_per_school</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;">NEMIS (~180k schools)</td>
              <td style="padding:2px 4px;border-bottom:1px solid #eee;"><b>State-level</b> (LGA harmonisation pending)</td></tr>
          <tr style="background:#fafafa;"><td style="padding:2px 4px;">nemis_jss/sss_schools, public/rural_pct</td>
              <td style="padding:2px 4px;">NEMIS</td>
              <td style="padding:2px 4px;"><b>State-level</b></td></tr>
        </table>
        <div style="margin-top:4px;color:#d63;font-size:10px;">
          <b>Note:</b> NBS and NEMIS features are state-level constants joined
          to all cells within a state — they do not add within-state variation.
          Within-state spatial pattern comes from RWI, nightlights, travel time,
          building density, distance signals, and DHS cluster proximity.
        </div>
      </details>

      <div style="margin-top:8px;color:#888;font-size:10px;border-top:1px solid #eee;padding-top:6px;">
        Research tool only — not official statistics.<br>
        UNICEF × Borealis AI · Nigeria pipeline · May 2026
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    return m


def run(cfg: dict | None = None) -> str | None:
    """
    Build the Nigeria comparison map and return the output path.

    Can be called programmatically from main.py (pass cfg) or run standalone.
    When cfg is provided, paths are resolved from cfg["paths"]; otherwise the
    script's own ROOT is used.
    """
    import logging
    log = logging.getLogger(__name__)

    # Resolve output path from cfg if available
    if cfg is not None:
        out_dir = Path(cfg["paths"].get("maps_dir", str(MAP_DIR)))
    else:
        out_dir = MAP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "nga_comparison_map.html"

    try:
        log.info("Building Nigeria comparison map …")
        gdf, cells_sample, state_df = load_data()
        log.info("  %d LGAs loaded", len(gdf))
        m = build_map(gdf, cells_sample=cells_sample, state_df=state_df)
        m.save(str(out))
        log.info("Comparison map saved: %s", out)
        return str(out)
    except Exception as exc:
        log.warning("Could not build comparison map (non-fatal): %s", exc)
        return None


def main():
    """Standalone entry point (python src/scripts/build_comparison_map.py)."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = run()
    if path:
        print(f"Saved: {path}")
        print("Open in your browser to explore all panels.")


if __name__ == "__main__":
    main()
