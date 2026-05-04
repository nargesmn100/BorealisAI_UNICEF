"""
Build an interactive multi-panel comparison map for Nigeria child deprivation.

Panels:
  1. MICS 2021 State Truth       — what we train on
  2. Ridge Model Prediction      — ML spatial disaggregation
  3. GAM Model Prediction        — additive model
  4. Absolute Error (Ridge)      — |prediction − truth| per state
  5. Uncertainty Band            — ridge_upper − ridge_lower
  6. NBS Monetary Poverty        — independent validation (NLSS 2019)

All panels shown at LGA level (775 LGAs) with hover tooltips.

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
# Load data
# ---------------------------------------------------------------------------
def load_data():
    gdf = gpd.read_file(ROOT / "Data/outputs/nga/maps/nga_lga_predictions.geojson")

    # Merge NBS state poverty
    nbs_path = ROOT / "Data/Nigeria/nbs/nga_nbs_state_poverty.csv"
    if nbs_path.exists():
        nbs = pd.read_csv(nbs_path)[["state", "poverty_headcount_pct"]]
        # harmonize state names
        nbs["state"] = nbs["state"].replace({
            "Federal Capital Territory": "FCT Abuja",
            "Cross River": "Cross River",
        })
        gdf = gdf.merge(nbs, on="state", how="left")
    else:
        gdf["poverty_headcount_pct"] = np.nan

    # Merge DHS zone deprivation
    dhs_path = ROOT / "Data/Nigeria/dhs/nga_dhs_zone_deprivation.csv"
    zone_map_path = ROOT / "src/utils/admin_mappings.py"

    # Absolute error
    gdf["ridge_abs_error"] = (gdf["ridge_moderate"] - gdf["mics_state_truth"]).abs()
    gdf["uncertainty_band"] = gdf["ridge_moderate_upper"] - gdf["ridge_moderate_lower"]

    return gdf


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
def build_map(gdf: gpd.GeoDataFrame) -> folium.Map:
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

    # ── Panel 3: GAM Prediction ────────────────────────────────────────────
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

    # ── Panel 4: Absolute Error ────────────────────────────────────────────
    err_max = float(gdf["ridge_abs_error"].quantile(0.95))
    cm4 = make_colormap(0, max(err_max, 5), ERROR_COLOURS, "Ridge Absolute Error (pp)")
    fg4 = folium.FeatureGroup(name="④ Ridge Absolute Error", show=False)
    for _, row in gdf.iterrows():
        val = row["ridge_abs_error"]
        color = cm4(np.clip(val, 0, err_max)) if pd.notna(val) else "#cccccc"
        tip = (
            f"<b>State:</b> {row['state']}<br>"
            f"<b>LGA:</b> {row['lga_name']}<br>"
            f"<b>|Error|:</b> {val:.1f} pp<br>"
            f"<b>Predicted:</b> {row['ridge_moderate']:.1f}%<br>"
            f"<b>Truth:</b> {row['mics_state_truth']:.1f}%"
        )
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda f, c=color: {"fillColor": c, "color": "#444", "weight": 0.4, "fillOpacity": 0.78},
            tooltip=folium.Tooltip(tip, sticky=True),
        ).add_to(fg4)
    fg4.add_to(m)
    cm4.add_to(m)

    # ── Panel 5: Uncertainty ───────────────────────────────────────────────
    unc_max = float(gdf["uncertainty_band"].quantile(0.95))
    cm5 = make_colormap(0, max(unc_max, 1), UNC_COLOURS, "Prediction Uncertainty (CI width, pp)")
    fg5 = folium.FeatureGroup(name="⑤ Uncertainty Band (CI width)", show=False)
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

    # ── Panel 6: NBS Monetary Poverty ─────────────────────────────────────
    if "poverty_headcount_pct" in gdf.columns and gdf["poverty_headcount_pct"].notna().any():
        nbs_max = float(gdf["poverty_headcount_pct"].max())
        cm6 = make_colormap(0, nbs_max, DEPRIV_COLOURS, "NBS NLSS 2019 — Monetary Poverty (%)")
        fg6 = folium.FeatureGroup(name="⑥ NBS NLSS 2019 Monetary Poverty", show=False)
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

    # ── Layer control + title ──────────────────────────────────────────────
    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    title_html = """
    <div style="
        position: fixed; top: 12px; left: 60px; z-index: 1000;
        background: white; padding: 10px 16px; border-radius: 8px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif; max-width: 320px;">
        <b style="font-size:14px;">Nigeria Child Deprivation — LGA Level</b><br>
        <span style="font-size:11px; color:#555;">
        Use the layer control (top right) to switch between:<br>
        ① MICS state truth &nbsp; ② Ridge prediction<br>
        ③ GAM prediction &nbsp;&nbsp; ④ Ridge error<br>
        ⑤ Uncertainty &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ⑥ NBS monetary poverty<br>
        <br>Hover: Ridge layer shows LGA-mean theme contributions when available.
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m


def main():
    print("Loading data …")
    gdf = load_data()
    print(f"  {len(gdf)} LGAs loaded")

    print("Building comparison map …")
    m = build_map(gdf)

    out = MAP_DIR / "nga_comparison_map.html"
    m.save(str(out))
    print(f"Saved: {out}")
    print("Open in your browser to explore all 6 panels.")


if __name__ == "__main__":
    main()
