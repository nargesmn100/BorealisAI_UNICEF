"""
Compare Ridge (and RWI) cell-level predictions to DHS cluster-level deprivation
at the nearest grid cell to each DHS GPS point.

This is *external* validation: DHS cluster deprivation was not used in training.
Coordinates are DHS-displaced (up to 2–5 km); interpret distances accordingly.

Outputs
-------
Data/outputs/nga/eval/dhs_gps_validation.csv
Data/outputs/nga/eval/dhs_gps_validation.txt
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Data/outputs/nga/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in km."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main():
    geo_path = ROOT / "Data/Nigeria/dhs/nga_dhs_cluster_deprivation_geo.geojson"
    pred_path = ROOT / "Data/outputs/nga/tables/nga_predictions.parquet"

    if not geo_path.is_file():
        raise SystemExit(f"Run first: python -m src.scripts.merge_dhs_gps\nMissing: {geo_path}")

    log.info("Loading DHS cluster points…")
    dhs = gpd.read_file(geo_path)
    if dhs.crs and dhs.crs.to_string() != "EPSG:4326":
        dhs = dhs.to_crs(4326)

    log.info("Loading grid predictions…")
    pred = pd.read_parquet(pred_path)
    tree = cKDTree(pred[["longitude", "latitude"]].values)
    pts = np.column_stack([dhs.geometry.x.values, dhs.geometry.y.values])
    _, idx = tree.query(pts, k=1)

    records = []
    for i, (_, row) in enumerate(dhs.iterrows()):
        j = int(idx[i])
        p = pred.iloc[j]
        d_km = haversine_km(
            p["longitude"], p["latitude"],
            float(dhs.geometry.x.iloc[i]), float(dhs.geometry.y.iloc[i]),
        )
        records.append({
            "cluster_id":         row["cluster_id"],
            "dhs_deprivation_index": row["deprivation_index"],
            "dhs_stunting_rate":  row.get("stunting_rate", np.nan),
            "adm1name":          row.get("ADM1NAME", ""),
            "urban_rural_dhs":   row.get("URBAN_RURA", ""),
            "nearest_cell_id":   int(p["cell_id"]),
            "dist_to_cell_km":   float(d_km) if not isinstance(d_km, np.ndarray) else float(d_km.item()),
            "mics_state_truth_moderate": p["moderate_prevalence"],
            "ridge_moderate":     p["ridge_moderate"],
            "rwi_moderate":       p.get("rwi_moderate", np.nan),
            "gam_moderate":       p.get("gam_moderate", np.nan),
        })

    val = pd.DataFrame(records)

    # DHS index is 0–1, predictions are 0–100
    val["dhs_deprivation_scaled"] = val["dhs_deprivation_index"] * 100.0

    ok = val["ridge_moderate"].notna() & val["dhs_deprivation_scaled"].notna()
    x, y = val.loc[ok, "dhs_deprivation_scaled"], val.loc[ok, "ridge_moderate"]
    rho_ridge, p_ridge = stats.spearmanr(x, y)
    r_pear, p_pear = stats.pearsonr(x, y)
    mae = np.abs(y - x).mean()

    ok_rwi = val["rwi_moderate"].notna() & val["dhs_deprivation_scaled"].notna()
    rho_rwi, p_rwi = stats.spearmanr(
        val.loc[ok_rwi, "dhs_deprivation_scaled"],
        val.loc[ok_rwi, "rwi_moderate"],
    )

    lines = [
        "=" * 72,
        "DHS GPS — Cluster vs nearest-grid prediction validation",
        "Nigeria DHS 2018 clusters × MICS-reconciled model outputs",
        "=" * 72,
        "",
        f"Valid DHS cluster points: {len(val)}",
        f"Mean distance from DHS point to grid cell centre: {val['dist_to_cell_km'].mean():.2f} km",
        f"Median distance: {val['dist_to_cell_km'].median():.2f} km",
        "",
        "DHS deprivation_index scaled ×100 to compare to model % on same visual scale.",
        "",
        f"Ridge: Spearman ρ = {rho_ridge:+.3f}  (p = {p_ridge:.4f})",
        f"Ridge: Pearson  r = {r_pear:+.3f}  (p = {p_pear:.4f})",
        f"Ridge: MAE      = {mae:.2f} pp  (DHS index×100 vs Ridge %)", 
        f"RWI:   Spearman ρ = {rho_rwi:+.3f}  (p = {p_rwi:.4f})",
        "",
        "NOTE: DHS index ≠ MICS moderate prevalence — different constructs.",
        "      Positive correlation = spatial co-movement of model with survey misery.",
        "=" * 72,
    ]
    report = "\n".join(lines)
    print(report)

    out_csv = OUT_DIR / "dhs_gps_validation.csv"
    out_txt = OUT_DIR / "dhs_gps_validation.txt"
    val.to_csv(out_csv, index=False)
    with open(out_txt, "w") as f:
        f.write(report)
    log.info("Saved: %s", out_csv)
    log.info("Saved: %s", out_txt)


if __name__ == "__main__":
    main()
