"""
Geospatial utility functions shared across pipeline steps.

Includes helpers for:
- raster windowed reads clipped to a bounding box
- point-in-raster sampling
- CRS validation
- SMOD class decoding
"""

import logging
import zipfile
import io
import tempfile
import os
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMOD class definitions
# ---------------------------------------------------------------------------
SMOD_CLASSES = {
    -200: "NoData",
    10: "Water",
    11: "Very Low Density Rural",
    12: "Low Density Rural",
    13: "Rural Cluster",
    21: "Suburban / Peri-urban",
    22: "Semi-dense Urban Cluster",
    23: "Dense Urban Cluster",
    30: "Urban Centre",
}

SMOD_URBAN_THRESHOLD = 21  # cells >= this are classified as urban


def decode_smod(value: int) -> str:
    """Return a human-readable label for a GHSL SMOD class value."""
    return SMOD_CLASSES.get(int(value), f"Unknown ({value})")


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def assert_crs_match(gdf1, gdf2, label1: str = "A", label2: str = "B") -> None:
    """
    Raise an AssertionError if two GeoDataFrames have different CRS.

    Parameters
    ----------
    gdf1, gdf2 : GeoDataFrame
    label1, label2 : str
        Descriptive names used in the error message.
    """
    if gdf1.crs != gdf2.crs:
        raise AssertionError(
            f"CRS mismatch between {label1} ({gdf1.crs}) and {label2} ({gdf2.crs}). "
            "Reproject before joining."
        )


def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to ``target_crs`` if it is not already in that CRS.

    Parameters
    ----------
    gdf : GeoDataFrame
    target_crs : str
        Target CRS string (e.g. "EPSG:4326").

    Returns
    -------
    GeoDataFrame
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS set. Cannot reproject.")
    if gdf.crs.to_epsg() != CRS.from_user_input(target_crs).to_epsg():
        logger.info("Reprojecting from %s to %s", gdf.crs, target_crs)
        gdf = gdf.to_crs(target_crs)
    return gdf


# ---------------------------------------------------------------------------
# Raster utilities
# ---------------------------------------------------------------------------

def open_raster_from_zip(
    zip_path: str, tif_name: str
) -> rasterio.DatasetReader:
    """
    Open a GeoTIFF stored inside a ZIP archive.

    For large global rasters (>200 MB), the file is extracted to a temporary
    directory on disk to avoid memory exhaustion.  For smaller files, it is
    loaded via rasterio MemoryFile.

    .. note::
        The caller is responsible for closing the returned dataset.
        The temporary directory (if created) is cleaned up automatically
        when the dataset is closed only if you use the context manager helper
        ``open_raster_from_zip_ctx`` instead.

    Parameters
    ----------
    zip_path : str
        Path to the ZIP file.
    tif_name : str
        Name of the .tif file inside the archive.

    Returns
    -------
    rasterio.DatasetReader
    """
    _MEMORY_THRESHOLD_BYTES = 200 * 1024 * 1024  # 200 MB

    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(tif_name)
        file_size = info.file_size

        if file_size <= _MEMORY_THRESHOLD_BYTES:
            # Small file — use in-memory approach
            tif_bytes = zf.read(tif_name)
            mem_file = rasterio.MemoryFile(tif_bytes)
            return mem_file.open()
        else:
            # Large file — extract to temp dir on disk
            tmpdir = tempfile.mkdtemp(prefix="borealis_raster_")
            extracted_path = zf.extract(tif_name, path=tmpdir)
            logger.debug(
                "Large raster (%d MB) extracted to temp path: %s",
                file_size // (1024 * 1024), extracted_path,
            )
            return rasterio.open(extracted_path)


def sample_raster_at_points(
    raster_path_or_dataset,
    lons: np.ndarray,
    lats: np.ndarray,
    band: int = 1,
    nodata_value: Optional[float] = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Sample a raster at a set of (lon, lat) point coordinates.

    Uses rasterio's ``sample`` method for efficient nearest-neighbour lookup.

    Parameters
    ----------
    raster_path_or_dataset : str or rasterio.DatasetReader
        Either a file path or an already-opened rasterio dataset.
    lons, lats : array-like
        WGS84 longitude and latitude arrays (same length).
    band : int
        Band number (1-indexed).
    nodata_value : float or None
        Value to treat as no-data.  If None, reads from raster metadata.
    fill_value : float
        Replacement value for no-data pixels.

    Returns
    -------
    np.ndarray
        Sampled values, shape (N,).
    """
    coords = list(zip(lons, lats))
    close_after = False

    if isinstance(raster_path_or_dataset, str):
        src = rasterio.open(raster_path_or_dataset)
        close_after = True
    else:
        src = raster_path_or_dataset

    try:
        if nodata_value is None:
            nodata_value = src.nodata

        # Reproject points to raster CRS if necessary
        raster_crs = src.crs
        if raster_crs and raster_crs.to_epsg() != 4326:
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            px, py = transformer.transform(np.array(lons), np.array(lats))
            coords = list(zip(px, py))

        values = np.array([v[band - 1] for v in src.sample(coords)])

        if nodata_value is not None:
            mask = values == nodata_value
            if np.isnan(nodata_value):
                mask = np.isnan(values)
            values = values.astype(float)
            values[mask] = fill_value

        # Replace any remaining large sentinel values
        values = values.astype(float)
        values[np.abs(values) > 1e9] = fill_value

        return values
    finally:
        if close_after:
            src.close()


def clip_raster_to_bbox(
    src: rasterio.DatasetReader,
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple:
    """
    Read a windowed portion of a raster corresponding to a bounding box.

    Works in the raster's native CRS (no reprojection).  The bbox should
    be supplied in the **raster's native CRS units**.

    Parameters
    ----------
    src : rasterio.DatasetReader
    west, south, east, north : float
        Bounding box in raster's native CRS.

    Returns
    -------
    (data, transform) : (np.ndarray shape (bands, rows, cols), Affine)
    """
    window = from_bounds(west, south, east, north, src.transform)
    data = src.read(window=window)
    transform = src.window_transform(window)
    return data, transform


def reproject_raster_to_wgs84(
    src: rasterio.DatasetReader,
    target_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.nearest,
) -> tuple:
    """
    Reproject a raster dataset to WGS84 (or another target CRS).

    Returns in-memory reprojected data and the new transform / CRS.

    Parameters
    ----------
    src : rasterio.DatasetReader
    target_crs : str
    resampling : rasterio.enums.Resampling

    Returns
    -------
    (data, transform, crs) : (np.ndarray, Affine, CRS)
    """
    dst_crs = CRS.from_user_input(target_crs)
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    data = np.empty((src.count, height, width), dtype=src.dtypes[0])
    for band_idx in range(src.count):
        reproject(
            source=rasterio.band(src, band_idx + 1),
            destination=data[band_idx],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return data, transform, dst_crs


# ---------------------------------------------------------------------------
# Spatial join helpers
# ---------------------------------------------------------------------------

def points_to_geodataframe(
    df: pd.DataFrame,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Convert a DataFrame with lon/lat columns to a GeoDataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    lon_col, lat_col : str
    crs : str

    Returns
    -------
    GeoDataFrame
    """
    geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)


def spatial_join_points_to_polygons(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    polygon_cols: list,
    how: str = "left",
) -> gpd.GeoDataFrame:
    """
    Spatial join points to polygons, keeping only selected polygon columns.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Point layer.
    polygons_gdf : GeoDataFrame
        Polygon layer with attributes to attach.
    polygon_cols : list of str
        Columns from polygons_gdf to carry into the result.
    how : str
        Join type (default "left" = keep all points).

    Returns
    -------
    GeoDataFrame
        Points enriched with polygon attributes.
    """
    assert_crs_match(points_gdf, polygons_gdf, "points", "polygons")

    poly_subset = polygons_gdf[["geometry"] + polygon_cols].copy()
    joined = gpd.sjoin(points_gdf, poly_subset, how=how, predicate="within")

    n_unmatched = joined[polygon_cols[0]].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            "%d points did not fall within any polygon and will have NaN "
            "for polygon attributes. Check bounding boxes / CRS alignment.",
            n_unmatched,
        )

    return joined.drop(columns=["index_right"], errors="ignore")
