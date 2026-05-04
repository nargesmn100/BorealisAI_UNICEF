"""
Explainability-first interactive map — Ridge theme dominance.

For each cell with a breakdown, determines the **dominant theme** (the
Ridge feature group with the largest |contribution| in β·z space) and
colours cells accordingly.  A second layer shows theme *direction*:
+/- contribution for the dominant group.

Reads
-----
  Data/outputs/nga/tables/nga_prediction_breakdown.csv
  (optional) Data/outputs/nga/maps/nga_lga_predictions.geojson   — LGA polygons background

Output
------
  Data/outputs/nga/maps/nga_explainability_map.html

Usage
-----
  python src/scripts/build_explainability_map.py
  python src/scripts/build_explainability_map.py --lga-background
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAPS_DIR = ROOT / "Data/outputs/nga/maps"
TABLES_DIR = ROOT / "Data/outputs/nga/tables"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import folium
    from branca.colormap import StepColormap
    import branca.colormap as cm
except ImportError:
    sys.exit("pip install folium branca")

# ── Theme palette (8 groups + 'other') ──────────────────────────────────────
THEME_COLOURS: dict[str, str] = {
    "wealth":          "#1f78b4",   # blue
    "urban_built":     "#33a02c",   # green
    "access_services": "#ff7f00",   # orange
    "nightlights":     "#ffd700",   # gold
    "health_mics":     "#e31a1c",   # red
    "education_mics":  "#6a3d9a",   # purple
    "hazards":         "#b15928",   # brown
    "dhs_cluster":     "#a6cee3",   # light-blue
    "other":           "#999999",   # grey
}

THEME_LABELS: dict[str, str] = {
    "wealth":          "Wealth (RWI / pop)",
    "urban_built":     "Urban / built-up",
    "access_services": "Access (travel / services)",
    "nightlights":     "Night-lights",
    "health_mics":     "Health utilization (MICS)",
    "education_mics":  "Education (MICS)",
    "hazards":         "Hazards (conflict / rain)",
    "dhs_cluster":     "DHS cluster signal",
    "other":           "Other",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dominant_theme(row: pd.Series, theme_cols: list[str]) -> tuple[str, float]:
    """Return (dominant_theme_slug, signed_value) for a row."""
    best_name, best_abs, best_val = "other", 0.0, 0.0
    for col in theme_cols:
        v = row.get(col)
        if pd.isna(v):
            continue
        slug = str(col).replace("ridge_theme__", "")
        if abs(float(v)) > best_abs:
            best_abs = abs(float(v))
            best_val = float(v)
            best_name = slug
    return best_name, best_val


def _popup_html(row: pd.Series, dominant: str, dominant_val: float, theme_cols: list[str]) -> str:
    parts = [
        f"<b>{row.get('subregion', '')} — cell {row.get('cell_id', '')}</b><br>",
        f"<b>Ridge pred:</b> {row.get('ridge_moderate', row.get('moderate_prevalence', 'N/A')):.1f}%<br>",
        f"<b>Dominant theme:</b> {THEME_LABELS.get(dominant, dominant)} ({dominant_val:+.2f})<br>",
        "<hr style='margin:4px 0'>",
        "<small><b>All themes (β·z):</b><br>",
    ]
    theme_vals = []
    for col in theme_cols:
        v = row.get(col)
        if pd.isna(v):
            continue
        slug = str(col).replace("ridge_theme__", "")
        theme_vals.append((abs(float(v)), slug, float(v)))
    theme_vals.sort(key=lambda x: -x[0])
    for _, slug, v in theme_vals[:6]:
        col_dot = f"<span style='color:{THEME_COLOURS.get(slug, '#999')};'>■</span> "
        parts.append(f"{col_dot}{THEME_LABELS.get(slug, slug)}: {v:+.2f}<br>")
    parts.append("</small>")
    return "".join(parts)


# ── Build map ────────────────────────────────────────────────────────────────

def build_map(brk: pd.DataFrame, lga_geojson: str | None = None) -> folium.Map:
    theme_cols = [c for c in brk.columns if c.startswith("ridge_theme__")]
    if not theme_cols:
        sys.exit("No ridge_theme__ columns in breakdown CSV. Run the pipeline first.")

    # Compute dominant theme per cell
    brk = brk.copy()
    dom = brk.apply(lambda r: _dominant_theme(r, theme_cols), axis=1)
    brk["_dom_theme"] = [d[0] for d in dom]
    brk["_dom_val"] = [d[1] for d in dom]

    center_lat = brk["latitude"].mean()
    center_lon = brk["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6,
                   tiles="CartoDB positron", control_scale=True)

    # Optional LGA polygon background
    if lga_geojson and os.path.isfile(lga_geojson):
        folium.GeoJson(
            lga_geojson,
            name="LGA boundaries",
            style_function=lambda f: {
                "color": "#555", "weight": 0.6, "fillOpacity": 0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["lga_name", "state", "ridge_moderate"],
                aliases=["LGA", "State", "Ridge (%)"],
            ),
        ).add_to(m)

    # ── Layer 1: Dominant theme (colour by theme) ─────────────────────────
    fg1 = folium.FeatureGroup(name="① Dominant Ridge theme", show=True)
    for _, row in brk.iterrows():
        lat, lon = row.get("latitude"), row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        dom_t = row["_dom_theme"]
        dom_v = row["_dom_val"]
        color = THEME_COLOURS.get(dom_t, "#999999")
        pop_html = _popup_html(row, dom_t, dom_v, theme_cols)
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=None,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            popup=folium.Popup(pop_html, max_width=340),
            tooltip=f"{THEME_LABELS.get(dom_t, dom_t)} ({dom_v:+.2f})",
        ).add_to(fg1)
    fg1.add_to(m)

    # ── Layer 2: Dominant theme direction (+/- contribution) ─────────────
    fg2 = folium.FeatureGroup(name="② Theme direction (+/−)", show=False)
    for _, row in brk.iterrows():
        lat, lon = row.get("latitude"), row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        dom_v = row["_dom_val"]
        color = "#d73027" if dom_v > 0 else "#2166ac"
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=None,
            fill=True,
            fill_color=color,
            fill_opacity=0.60,
            tooltip=f"{row['_dom_theme']} {dom_v:+.2f}",
        ).add_to(fg2)
    fg2.add_to(m)

    # ── Layer 3: DHS cluster signal strength ─────────────────────────────
    dhs_col = next((c for c in theme_cols if "dhs" in c), None)
    if dhs_col:
        fg3 = folium.FeatureGroup(name="③ DHS cluster contribution", show=False)
        vals = brk[dhs_col].dropna()
        vmin_d, vmax_d = vals.quantile(0.02), vals.quantile(0.98)
        cmap_d = cm.LinearColormap(
            ["#2166ac", "#f7f7f7", "#d73027"],
            vmin=vmin_d, vmax=vmax_d,
            caption="DHS cluster β·z",
        )
        cmap_d.add_to(m)
        for _, row in brk.iterrows():
            lat, lon = row.get("latitude"), row.get("longitude")
            v = row.get(dhs_col)
            if pd.isna(lat) or pd.isna(lon) or pd.isna(v):
                continue
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=None,
                fill=True,
                fill_color=cmap_d(np.clip(float(v), vmin_d, vmax_d)),
                fill_opacity=0.65,
                tooltip=f"DHS β·z: {float(v):+.2f}",
            ).add_to(fg3)
        fg3.add_to(m)

    # ── Legend HTML ───────────────────────────────────────────────────────
    theme_legend_rows = "".join(
        f"<span style='color:{THEME_COLOURS[k]};font-size:14px;'>■</span> {THEME_LABELS[k]}<br>"
        for k in THEME_COLOURS
    )
    legend_html = f"""
    <div style="position:fixed;top:12px;left:60px;z-index:1000;
                background:white;padding:10px 14px;border-radius:8px;
                box-shadow:2px 2px 8px rgba(0,0,0,.3);
                font-family:Arial,sans-serif;max-width:280px;font-size:11px;">
      <b style="font-size:13px;">Ridge Feature Theme — Dominance</b><br>
      <span style="color:#555;font-size:10px;">
        Circle colour = feature group with largest |β·z| per cell.<br>
        Layer ②: red = poverty-increasing, blue = decreasing.<br>
        Click any circle for full theme breakdown.
      </span><br>
      <hr style="margin:4px 0">
      {theme_legend_rows}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False, position="topright").add_to(m)
    return m


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build Ridge theme-dominance explainability map.")
    parser.add_argument("--lga-background", action="store_true",
                        help="Overlay LGA polygon boundaries.")
    parser.add_argument("--breakdown-csv", default=str(TABLES_DIR / "nga_prediction_breakdown.csv"),
                        help="Path to nga_prediction_breakdown.csv.")
    parser.add_argument("--out", default=str(MAPS_DIR / "nga_explainability_map.html"),
                        help="Output HTML path.")
    args = parser.parse_args()

    if not os.path.isfile(args.breakdown_csv):
        sys.exit(
            f"Breakdown CSV not found: {args.breakdown_csv}\n"
            "Run: python main.py --country nga --skip-gbm --skip-gam --skip-wsnn"
        )

    print(f"Loading breakdown: {args.breakdown_csv}")
    brk = pd.read_csv(args.breakdown_csv)
    print(f"  {len(brk):,} cells loaded")

    lga_geojson = str(MAPS_DIR / "nga_lga_predictions.geojson") if args.lga_background else None

    print("Building explainability map…")
    m = build_map(brk, lga_geojson=lga_geojson)
    m.save(args.out)
    print(f"Saved: {args.out}")
    print("Open in a browser.  Use layer control (top right) to switch views.")


if __name__ == "__main__":
    main()
