"""
Merge DHS 2018 cluster deprivation table with official DHS GPS shapefile (NGGE7BFL).

Join key: cluster_id (from microdata v001 / hv001) == DHSCLUST in the shapefile.

Outputs
-------
Data/Nigeria/dhs/nga_dhs_cluster_deprivation_geo.csv
Data/Nigeria/dhs/nga_dhs_cluster_deprivation_geo.geojson
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Data/Nigeria/dhs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Shapefile may live under Data/Nigeria/dhs/raw/, repo root, or Data/Nigeria/dhs/
_SHP_CANDIDATES = [
    ROOT / "Data" / "Nigeria" / "dhs" / "raw" / "NGGE7BFL" / "NGGE7BFL.shp",
    ROOT / "NGGE7BFL" / "NGGE7BFL.shp",
    ROOT / "Data" / "Nigeria" / "dhs" / "NGGE7BFL.shp",
]


def find_shp() -> Path:
    for p in _SHP_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "NGGE7BFL.shp not found. Place the DHS GPS shapefile at one of:\n  "
        + "\n  ".join(str(p) for p in _SHP_CANDIDATES)
    )


def main():
    cluster_csv = OUT_DIR / "nga_dhs_cluster_deprivation.csv"
    if not cluster_csv.is_file():
        raise SystemExit(
            f"Run first: python -m src.scripts.process_dhs\nMissing: {cluster_csv}"
        )

    shp_path = find_shp()
    log.info("Loading GPS shapefile: %s", shp_path)
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(4326)

    # DHSCLUST matches v001 / cluster_id
    gdf = gdf.rename(columns={"DHSCLUST": "cluster_id"})
    gdf["cluster_id"] = pd.to_numeric(gdf["cluster_id"], errors="coerce").astype("Int64")

    keep_meta = [
        "cluster_id", "DHSID", "ADM1DHS", "ADM1NAME", "DHSREGCO", "DHSREGNA",
        "SOURCE", "URBAN_RURA", "LATNUM", "LONGNUM", "ALT_DEM", "geometry",
    ]
    gdf = gdf[[c for c in keep_meta if c in gdf.columns]]

    clusters = pd.read_csv(cluster_csv)
    clusters["cluster_id"] = clusters["cluster_id"].astype(int)

    merged = clusters.merge(
        gdf.drop(columns=["geometry"], errors="ignore"),
        on="cluster_id",
        how="left",
        validate="one_to_one",
    )

    n_bad = ((merged["LATNUM"].fillna(0) == 0) & (merged["LONGNUM"].fillna(0) == 0)).sum()
    if n_bad:
        log.warning("Clusters with 0,0 coordinates (MIS / missing): %d", n_bad)

    # Geometry from merged row — rebuild GeoDataFrame for file export
    merged_gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["LONGNUM"], merged["LATNUM"], crs="EPSG:4326"),
    )
    # Drop invalid points
    valid = (merged_gdf["LATNUM"].notna()) & (merged_gdf["LONGNUM"].notna())
    # DHS uses 0,0 for unlocated clusters
    valid &= ~((merged_gdf["LATNUM"] == 0) & (merged_gdf["LONGNUM"] == 0))
    merged_gdf = merged_gdf[valid].copy()

    out_csv = OUT_DIR / "nga_dhs_cluster_deprivation_geo.csv"
    out_gj = OUT_DIR / "nga_dhs_cluster_deprivation_geo.geojson"

    merged_gdf.drop(columns=["geometry"]).to_csv(out_csv, index=False)
    merged_gdf.to_file(out_gj, driver="GeoJSON")

    log.info("Saved: %s (%d clusters with valid coordinates)", out_csv, len(merged_gdf))
    log.info("Saved: %s", out_gj)
    log.info("Matched all deprivation rows: %d / %d", len(merged), len(clusters))


if __name__ == "__main__":
    main()
