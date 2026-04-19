"""
Process GHSL Built-Up Surface data for Nigeria

Downloads / processes the Global Human Settlement Layer (GHSL) built-up surface
fraction (GHS_BUILT_S R2023A) and samples it onto the Nigeria RWI grid.

MANUAL DOWNLOAD REQUIRED
-------------------------
1. Open in browser:
   https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/
   GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss/V1-0/
   GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0.zip
2. Unzip and place the .tif inside:
   Data/Nigeria/features/ghsl/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif

WHAT THIS ADDS
--------------
  ghsl_built_m2     : built-up surface area per grid cell (m²)
  ghsl_built_frac   : fraction of grid cell covered by buildings (0–1)
  log_ghsl_built    : log1p(ghsl_built_m2), for modeling

Usage
-----
    python -m src.scripts.process_ghsl

Outputs
-------
  Data/Nigeria/features/nga_new_features.parquet
      (merged into existing new_features or created fresh)
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
GHSL_TIF   = "Data/Nigeria/features/ghsl/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
RWI_GRID   = "Data/Nigeria/nga_relative_wealth_index.csv"
OUT_FEATS  = "Data/Nigeria/features/nga_new_features.parquet"
NGA_BBOX   = (2.67, 4.26, 14.68, 13.87)   # (lon_min, lat_min, lon_max, lat_max)


def sample_ghsl(tif_path: str, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Sample GHSL raster at given lon/lat coordinates using bilinear interpolation.
    Returns array of built-up surface area values (m² per 3-arc-second cell).
    Missing / nodata values returned as NaN.
    """
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        xs, ys = lons, lats
        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        h, w = src.height, src.width
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

        data = src.read(1)
        values = np.full(len(lons), np.nan, dtype=np.float32)
        values[valid] = data[rows[valid], cols[valid]].astype(np.float32)

        if nodata is not None:
            values[values == nodata] = np.nan

    return values


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # Check GHSL file exists
    if not os.path.isfile(GHSL_TIF):
        logger.error(
            "GHSL TIF not found: %s\n"
            "Please download manually — see docstring at top of this file.\n"
            "Direct URL:\n"
            "  https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_BUILT_S_GLOBE_R2023A/"
            "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss/V1-0/"
            "GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip",
            GHSL_TIF,
        )
        sys.exit(1)

    # Load RWI grid
    logger.info("Loading RWI grid: %s", RWI_GRID)
    grid = pd.read_csv(RWI_GRID)
    lon_min, lat_min, lon_max, lat_max = NGA_BBOX
    grid = grid[
        (grid["longitude"] >= lon_min) & (grid["longitude"] <= lon_max) &
        (grid["latitude"]  >= lat_min) & (grid["latitude"]  <= lat_max)
    ].reset_index(drop=True)
    logger.info("Grid cells: %d", len(grid))

    # Sample GHSL
    logger.info("Sampling GHSL built-up surface from: %s", GHSL_TIF)
    ghsl_vals = sample_ghsl(GHSL_TIF, grid["longitude"].values, grid["latitude"].values)

    n_valid = np.isfinite(ghsl_vals).sum()
    logger.info(
        "GHSL sampled: %d / %d cells with data. Mean=%.1f m², Max=%.1f m²",
        n_valid, len(grid), np.nanmean(ghsl_vals), np.nanmax(ghsl_vals),
    )

    # Compute derived features
    # GHS_BUILT_S at 30-arc-second (~900m) — value is m² of built-up surface per cell
    # Cell area at equator ≈ 900m × 900m = 810000 m²
    cell_area_m2 = 810000.0
    ghsl_frac  = np.clip(ghsl_vals / cell_area_m2, 0, 1)
    log_ghsl   = np.log1p(ghsl_vals)

    # Build feature DataFrame
    feat_df = pd.DataFrame({
        "longitude":       grid["longitude"].round(6),
        "latitude":        grid["latitude"].round(6),
        "ghsl_built_m2":   np.where(np.isfinite(ghsl_vals), ghsl_vals, 0.0).astype(np.float32),
        "ghsl_built_frac": np.where(np.isfinite(ghsl_frac),  ghsl_frac, 0.0).astype(np.float32),
        "log_ghsl_built":  np.where(np.isfinite(log_ghsl),   log_ghsl,  0.0).astype(np.float32),
    })

    # Merge into existing new_features parquet (or save fresh)
    if os.path.isfile(OUT_FEATS):
        existing = pd.read_parquet(OUT_FEATS)
        for c in ["longitude", "latitude"]:
            existing[c] = existing[c].round(6)
            feat_df[c]  = feat_df[c].round(6)
        # Drop old GHSL columns if re-running
        for c in ["ghsl_built_m2", "ghsl_built_frac", "log_ghsl_built"]:
            if c in existing.columns:
                existing = existing.drop(columns=[c])
        merged = existing.merge(feat_df, on=["longitude", "latitude"], how="left")
        logger.info("Merged GHSL features into existing feature table (%d rows).", len(merged))
    else:
        os.makedirs(os.path.dirname(OUT_FEATS), exist_ok=True)
        merged = feat_df
        logger.info("Created new feature table with GHSL features (%d rows).", len(merged))

    merged.to_parquet(OUT_FEATS, index=False)
    logger.info("Saved: %s", OUT_FEATS)
    logger.info(
        "New columns: ghsl_built_m2, ghsl_built_frac, log_ghsl_built\n"
        "Next steps:\n"
        "  1. Add 'ghsl_built_frac' and 'log_ghsl_built' to modeling.features in config_nga.yaml\n"
        "  2. Re-run: python main.py --country nga --phase models"
    )


if __name__ == "__main__":
    main()
