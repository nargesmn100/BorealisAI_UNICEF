"""
Build multi-panel Per-Dimension Deprivation Map (LGA Level)
===========================================================

Creates an interactive HTML map with one Leaflet panel per Kyriaki dimension
plus a composite panel — LGA polygons, population-weighted moderate %.

Each panel shows predicted **moderate deprivation prevalence** for that
dimension at LGA level (see generated HTML “How to read this map”).

Usage
-----
    python src/scripts/build_dimension_map.py --country nga
    python src/scripts/build_dimension_map.py --country nga --output-path Data/outputs/nga/maps/my_map.html

Output
------
    Data/outputs/nga/maps/nga_dimension_comparison_map.html
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from src.utils.config_loader import load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Panel definitions
# ---------------------------------------------------------------------------

PANELS = [
    {"col": "shelter_moderate",       "title": "Shelter",              "subtitle": "≥3 persons / sleeping room"},
    {"col": "sanitation_moderate",    "title": "Sanitation",           "subtitle": "Improved but shared toilet"},
    {"col": "water_moderate",         "title": "Water",                "subtitle": "Improved but >30 min roundtrip"},
    {"col": "nutrition_moderate",     "title": "Nutrition",            "subtitle": "DHS HAZ stunting (moderate)"},
    {"col": "edu_5_14_moderate",      "title": "Education 5–14",       "subtitle": "Not currently attending school"},
    {"col": "edu_15_17_moderate",     "title": "Education 15–17",      "subtitle": "Not in secondary / no sec. completion"},
    {"col": "health_moderate",        "title": "Health 12–35m",        "subtitle": "Missing DPT1–3 or measles vaccine"},
    {"col": "health_36_59_moderate",  "title": "Health 36–59m",        "subtitle": "ARI/fever, no professional care-seeking"},
    {"col": "ridge_moderate",         "title": "Composite (Ridge)",    "subtitle": "Composite moderate deprivation (all dims)"},
]

# Colour scheme per panel (index → CSS gradient end colour)
PANEL_COLOURS = [
    "#7b2d8b",  # shelter      — purple
    "#1a6632",  # sanitation   — dark green
    "#1565c0",  # water        — dark blue
    "#e65100",  # nutrition    — deep orange
    "#b71c1c",  # edu 5-14     — dark red
    "#880e4f",  # edu 15-17    — dark pink
    "#006064",  # health 12-35 — dark teal
    "#004d40",  # health 36-59 — dark green-teal
    "#37474f",  # composite    — dark grey
]


# ---------------------------------------------------------------------------
# HTML/CSS helpers
# ---------------------------------------------------------------------------

def _css_gradient(end_colour: str) -> str:
    return f"linear-gradient(135deg, #fff7e6 0%, {end_colour} 100%)"


def _colour_for_value(val: float, vmin: float, vmax: float, end_colour: str) -> str:
    """Interpolate from #fffde7 (near-white yellow) to end_colour."""
    if vmax == vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))

    # Parse end_colour hex
    h = end_colour.lstrip("#")
    r2, g2, b2 = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r1, g1, b1 = 255, 253, 231  # near-white

    r = int(r1 + t * (r2 - r1))
    g = int(g1 + t * (g2 - g1))
    b = int(b1 + t * (b2 - b1))
    return f"rgb({r},{g},{b})"


def _make_legend(vmin: float, vmax: float, end_colour: str, title: str) -> str:
    stops = [_colour_for_value(vmin + i * (vmax - vmin) / 4, vmin, vmax, end_colour)
             for i in range(5)]
    gradient = ", ".join(stops)
    return f"""
    <div class="legend">
      <div class="legend-title">{title}</div>
      <div class="legend-bar" style="background: linear-gradient(to right, {gradient});"></div>
      <div class="legend-labels">
        <span>{vmin:.0f}%</span>
        <span>{(vmin + vmax) / 2:.0f}%</span>
        <span>{vmax:.0f}%</span>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# GeoJSON feature builder
# ---------------------------------------------------------------------------

def _lga_geojson_feature(row: dict, col: str, vmin: float, vmax: float, end_colour: str,
                          geometry_json: str) -> str:
    val = row.get(col)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        fill = "#e0e0e0"
        label = "N/A"
    else:
        fill = _colour_for_value(float(val), vmin, vmax, end_colour)
        label = f"{val:.1f}%"
    return fill, label


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dimension_map(cfg: dict, output_path: str | None = None) -> str:
    """Build and save the 7-panel dimension comparison map. Returns output path."""
    tables_dir = cfg["paths"]["tables_dir"]
    maps_dir = cfg["paths"]["maps_dir"]
    prefix = cfg.get("country", {}).get("output_prefix", "nga")

    lga_csv = os.path.join(tables_dir, f"{prefix}_lga_predictions.csv")
    lga_geojson = os.path.join(maps_dir, f"{prefix}_lga_predictions.geojson")

    if not os.path.isfile(lga_csv):
        raise FileNotFoundError(
            f"LGA predictions not found: {lga_csv}\n"
            "Run: python main.py --country nga  (or re-run phase_outputs)"
        )

    lga = pd.read_csv(lga_csv)
    logger.info("LGA data loaded: %d LGAs × %d columns", *lga.shape)

    # Check which dimension columns exist
    available_panels = [p for p in PANELS if p["col"] in lga.columns]
    missing = [p["col"] for p in PANELS if p["col"] not in lga.columns]
    if missing:
        logger.warning("Missing columns (will show grey): %s", missing)

    if output_path is None:
        output_path = os.path.join(maps_dir, f"{prefix}_dimension_comparison_map.html")

    # ── Load GeoJSON for polygon geometries ──
    try:
        import json
        with open(lga_geojson) as f:
            geo = json.load(f)
        features = {
            feat["properties"].get("lga_id"): feat["geometry"]
            for feat in geo.get("features", [])
        }
        logger.info("GeoJSON features loaded: %d", len(features))
    except Exception as e:
        logger.warning("Could not load GeoJSON: %s — using point map fallback", e)
        features = {}

    # ── Compute per-column stats ──
    stats = {}
    for panel in PANELS:
        col = panel["col"]
        if col in lga.columns:
            vals = lga[col].dropna()
            stats[col] = {
                "vmin": float(vals.quantile(0.02)),
                "vmax": float(vals.quantile(0.98)),
            }
        else:
            stats[col] = {"vmin": 0, "vmax": 100}

    # ── Build per-panel Leaflet layers as JS variable arrays ──
    # Each panel = array of {lga_id, state, lga_name, fill, val, popup_html}
    panel_js_arrays = []
    for idx, panel in enumerate(PANELS):
        col = panel["col"]
        colour = PANEL_COLOURS[idx % len(PANEL_COLOURS)]
        vmin = stats[col]["vmin"]
        vmax = stats[col]["vmax"]

        entries = []
        for _, row in lga.iterrows():
            lid = row.get("lga_id", "")
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and np.isnan(float(val if val is not None else "nan"))):
                fill = "#e0e0e0"
                val_str = "N/A"
            else:
                fill = _colour_for_value(float(val), vmin, vmax, colour)
                val_str = f"{float(val):.1f}%"

            pop = int(row.get("total_population", 0) or 0)
            state = str(row.get("state", ""))
            lga_name = str(row.get("lga_name", ""))

            # Mini popup with all dimensions for this LGA
            dim_rows = ""
            for p in PANELS:
                v = row.get(p["col"])
                dim_rows += (
                    f'<tr><td>{p["title"]}</td>'
                    f'<td style="text-align:right;font-weight:bold">'
                    f'{v:.1f}%</td></tr>'
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                    else f'<tr><td>{p["title"]}</td><td>N/A</td></tr>'
                )

            popup = (
                f'<div style="font-family:sans-serif;font-size:12px;min-width:200px">'
                f'<b>{lga_name}</b> — {state}<br>'
                f'Population: {pop:,}<br><hr style="margin:4px 0">'
                f'<table style="width:100%">{dim_rows}</table></div>'
            )
            entries.append({
                "id": lid,
                "fill": fill,
                "val": val_str,
                "popup": popup,
            })

        panel_js_arrays.append(entries)

    # ── Build GeoJSON JS object for Leaflet ──
    geojson_js = "null"
    if features:
        import json
        geojson_features = []
        for _, row in lga.iterrows():
            lid = row.get("lga_id", "")
            geom = features.get(lid)
            if geom is None:
                continue
            props = {"lga_id": lid}
            geojson_features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": props,
            })
        geojson_js = json.dumps({"type": "FeatureCollection", "features": geojson_features})

    # ── Build full HTML ──
    panels_meta_js = []
    for idx, panel in enumerate(PANELS):
        col = panel["col"]
        colour = PANEL_COLOURS[idx % len(PANEL_COLOURS)]
        vmin = stats[col]["vmin"]
        vmax = stats[col]["vmax"]
        panels_meta_js.append(
            f'{{"col":"{col}","title":"{panel["title"]}","subtitle":"{panel["subtitle"]}",'
            f'"colour":"{colour}","vmin":{vmin:.1f},"vmax":{vmax:.1f}}}'
        )

    panels_data_js = []
    for entries in panel_js_arrays:
        import json
        panels_data_js.append(json.dumps(entries))

    # Build legend HTML per panel
    legends_html = ""
    for idx, panel in enumerate(PANELS):
        col = panel["col"]
        colour = PANEL_COLOURS[idx % len(PANEL_COLOURS)]
        vmin = stats[col]["vmin"]
        vmax = stats[col]["vmax"]
        legends_html += _make_legend(vmin, vmax, colour, panel["title"])

    n_panels = len(PANELS)
    cols_per_row = 4
    panel_grid = ""
    for idx in range(n_panels):
        panel = PANELS[idx]
        colour = PANEL_COLOURS[idx % len(PANEL_COLOURS)]
        panel_grid += f"""
        <div class="panel-card" id="card-{idx}">
          <div class="panel-header" style="background:{colour}">
            <span class="panel-title">{panel['title']}</span><br>
            <span class="panel-subtitle">{panel['subtitle']}</span>
          </div>
          <div id="map-{idx}" class="panel-map"></div>
          <div id="legend-{idx}" class="panel-legend"></div>
        </div>"""

    # Embedded documentation for stakeholders opening the HTML standalone.
    doc_section = """
<section class="map-docs" aria-label="How to read this map">
  <div class="map-docs-inner">
    <h2>How to read this map</h2>

    <h3>What does the percentage mean?</h3>
    <p>
      Each panel shows a <strong>predicted prevalence (%)</strong> of
      <strong>moderate deprivation</strong> on <strong>that dimension only</strong>,
      for <strong>one Local Government Area (LGA)</strong>.
      The value is a <strong>rate</strong>: “what share of children or youth in the
      relevant age band meet this dimension’s Kyriaki rule?” — not a score for one child,
      and not a dollar poverty line.
    </p>
    <div class="example">
      <strong>Example — Education 5–14 shows 60%:</strong>
      The model estimates that <strong>60% of children aged 5–14</strong> in that LGA are
      <strong>moderately deprived on education</strong> under this survey definition
      (here: not currently attending school). Another panel (e.g. Shelter or Nutrition)
      uses a different rule and age range — percentages are <strong>not comparable</strong>
      across panels as “severity rank.”
    </div>
    <p>
      Numbers come from <strong>Ridge models</strong> trained on <strong>state-level</strong>
      survey targets (MICS / DHS where used), then <strong>reconciled</strong> so each state’s
      population-weighted average matches its official target for that dimension.
      LGAs are <strong>population-weighted averages</strong> of finer grid cells (~2.4 km spacing).
    </p>

    <h3>What do the colours mean?</h3>
    <ul>
      <li><strong>Each panel has its own colour ramp</strong> (see the coloured panel title bar):
      pale cream / yellow = <strong>lower</strong> predicted prevalence in that LGA
      <em>relative to other LGAs in Nigeria on that same dimension</em>;
      rich hue (purple, green, blue, etc.) = <strong>higher</strong> prevalence.</li>
      <li>The legend min/max are set from the <strong>2nd–98th percentile</strong> of LGA values
      <em>for that panel</em> so most maps have visible contrast (the scale is not always 0–100%).</li>
      <li><strong>Grey</strong> fill = no prediction available for that LGA on that layer.</li>
    </ul>

    <h3>Composite panel</h3>
    <p>
      <strong>Composite (Ridge)</strong> is <strong>not</strong> the sum of the dimension panels.
      It shows <strong>multidimensional child deprivation</strong> under the MICS composite definition
      (several domains combined; see project docs). Single-dimension panels answer separate policy questions
      (water, education, nutrition, etc.).
    </p>
    <p style="margin-bottom:0;font-size:0.82rem;color:#555;">
      <strong>Disclaimer:</strong> Research prototype — not official government statistics.
    </p>
  </div>
</section>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nigeria — Per-Dimension Deprivation Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9; }}
  header {{
    background: #1a237e; color: white; padding: 16px 24px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  header h1 {{ margin:0; font-size:1.3rem; font-weight:600; }}
  header p  {{ margin:4px 0 0; font-size:0.85rem; opacity:0.85; }}
  .header-meta {{ text-align:right; font-size:0.8rem; opacity:0.8; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat({cols_per_row}, 1fr);
    gap: 12px;
    padding: 16px;
    max-width: 1600px;
    margin: 0 auto;
  }}
  .panel-card {{
    background: white;
    border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.12);
    overflow: hidden;
  }}
  .panel-header {{
    color: white; padding: 10px 14px;
  }}
  .panel-title {{ font-size:0.95rem; font-weight:600; }}
  .panel-subtitle {{ font-size:0.73rem; opacity:0.9; }}
  .panel-map {{ height: 260px; }}
  .panel-legend {{
    padding: 6px 10px 8px;
    font-size: 0.72rem;
  }}
  .legend-bar {{
    height: 10px; border-radius: 4px; margin: 4px 0 2px;
  }}
  .legend-labels {{ display:flex; justify-content:space-between; color:#555; }}
  footer {{
    text-align: center; padding: 12px; font-size: 0.78rem;
    color: #777; border-top: 1px solid #e0e0e0; margin-top: 8px;
  }}
  @media (max-width:1100px) {{ .grid {{ grid-template-columns: repeat(2,1fr); }} }}
  @media (max-width:600px)  {{ .grid {{ grid-template-columns: 1fr; }} }}
  .map-docs {{
    max-width: 1600px;
    margin: 12px auto 4px;
    padding: 0 16px;
  }}
  .map-docs-inner {{
    background: #fff;
    border: 1px solid #c5cae9;
    border-radius: 8px;
    padding: 14px 18px 16px;
    font-size: 0.88rem;
    line-height: 1.5;
    color: #333;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .map-docs-inner h2 {{
    margin: 0 0 10px 0;
    font-size: 1.05rem;
    color: #1a237e;
    font-weight: 600;
  }}
  .map-docs-inner h3 {{
    margin: 12px 0 6px 0;
    font-size: 0.92rem;
    color: #3949ab;
    font-weight: 600;
  }}
  .map-docs-inner p {{ margin: 6px 0; }}
  .map-docs-inner ul {{ margin: 6px 0 6px 1.1rem; padding: 0; }}
  .map-docs-inner li {{ margin: 4px 0; }}
  .map-docs-inner .example {{
    background: #fff8e1;
    border-left: 4px solid #ff8f00;
    padding: 8px 12px;
    margin: 10px 0;
    font-size: 0.84rem;
  }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Nigeria — Per-Dimension Child Deprivation</h1>
    <p>LGA-level predictions (Ridge regression, Kyriaki specification · May 2026)</p>
  </div>
  <div class="header-meta">
    {n_panels} panels · 775 LGAs · ~103k grid cells<br>
    Click any LGA for full breakdown
  </div>
</header>

{doc_section}

<div class="grid">
{panel_grid}
</div>

<footer>
  UNICEF × RBC Borealis AI · Research prototype · Not official statistics ·
  Colours = relative prevalence within each panel · Grey = no data · See box above for % meaning
</footer>

<script>
const GEOJSON = {geojson_js};
const PANELS_META = [{",".join(panels_meta_js)}];
const PANELS_DATA = [
  {("," + chr(10) + "  ").join(panels_data_js)}
];

function hexToRgb(hex) {{
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return {{r,g,b}};
}}

function interpolateColour(t, endHex) {{
  const {{r:r2,g:g2,b:b2}} = hexToRgb(endHex);
  const r1=255,g1=253,b1=231;
  return `rgb(${{Math.round(r1+t*(r2-r1))}},${{Math.round(g1+t*(g2-g1))}},${{Math.round(b1+t*(b2-b1))}})`;
}}

function makeLegend(container, meta) {{
  const stops = [0,0.25,0.5,0.75,1].map(t => interpolateColour(t, meta.colour));
  container.innerHTML = `
    <div style="font-size:0.7rem;color:#444;margin-bottom:2px">
      ${{meta.title}} moderate %
    </div>
    <div class="legend-bar" style="background:linear-gradient(to right,${{stops.join(',')}})"></div>
    <div class="legend-labels">
      <span>${{meta.vmin.toFixed(0)}}%</span>
      <span>${{((meta.vmin+meta.vmax)/2).toFixed(0)}}%</span>
      <span>${{meta.vmax.toFixed(0)}}%</span>
    </div>`;
}}

PANELS_META.forEach((meta, idx) => {{
  const mapDiv = document.getElementById(`map-${{idx}}`);
  const legendDiv = document.getElementById(`legend-${{idx}}`);
  makeLegend(legendDiv, meta);

  const map = L.map(mapDiv, {{
    center: [9.0, 8.0], zoom: 5,
    zoomControl: true, attributionControl: false,
  }});

  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    opacity: 0.5, maxZoom: 18,
  }}).addTo(map);

  const entries = PANELS_DATA[idx];
  const byId = {{}};
  entries.forEach(e => byId[e.id] = e);

  if (GEOJSON) {{
    L.geoJSON(GEOJSON, {{
      style: feature => {{
        const e = byId[feature.properties.lga_id];
        return {{
          fillColor: e ? e.fill : '#e0e0e0',
          weight: 0.5, color: '#fff', fillOpacity: 0.85,
        }};
      }},
      onEachFeature: (feature, layer) => {{
        const e = byId[feature.properties.lga_id];
        if (e) {{
          layer.bindPopup(e.popup, {{maxWidth: 260}});
          layer.on('mouseover', function() {{
            this.setStyle({{weight:2, color:'#333'}});
          }});
          layer.on('mouseout', function() {{
            this.setStyle({{weight:0.5, color:'#fff'}});
          }});
        }}
      }},
    }}).addTo(map);
  }}
}});
</script>
</body>
</html>"""

    os.makedirs(maps_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / 1e6
    logger.info("Dimension comparison map saved: %s (%.1f MB)", output_path, size_mb)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build multi-panel LGA-level dimension + composite comparison map."
    )
    parser.add_argument("--country", default="nga")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config_path = args.config or f"config/config_{args.country}.yaml"
    cfg = load_config(config_path)
    setup_logging(cfg)

    logger.info("Building dimension comparison map for %s...", args.country.upper())
    out = build_dimension_map(cfg, output_path=args.output_path)
    logger.info("Done: %s", out)


if __name__ == "__main__":
    main()
