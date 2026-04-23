"""
Download Geospatial Data for Nigeria Pipeline
==============================================

Helper functions to download required geospatial files (GADM boundaries,
WorldPop population, VIIRS nighttime lights, GHSL building density) if they
are not already present on disk.

All downloads are placed in Data/Nigeria/ by default.  These functions are
called automatically during the data phase if the expected files are missing.

Usage
-----
    from src.data.download_geospatial import ensure_geospatial_data
    ensure_geospatial_data(cfg)  # downloads any missing files
"""

import logging
import os
import urllib.request
import zipfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest_path: str, description: str = "") -> None:
    """Download a file from URL to dest_path with progress reporting."""
    if os.path.isfile(dest_path):
        logger.info("Already exists: %s", dest_path)
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info("Downloading %s from %s ...", description, url)

    try:
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 8192
        downloaded = 0
        last_pct_logged = -10  # so first log happens at 0%

        if total_size > 0:
            logger.info("  Total size: %.1f MB", total_size / (1024 * 1024))

        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    pct = int(downloaded * 100 / total_size)
                    if pct >= last_pct_logged + 10:
                        logger.info(
                            "  %s: %d%% (%.1f / %.1f MB)",
                            description or "download",
                            pct,
                            downloaded / (1024 * 1024),
                            total_size / (1024 * 1024),
                        )
                        last_pct_logged = pct

        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info("Downloaded %s (%.1f MB)", dest_path, size_mb)
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        if os.path.isfile(dest_path):
            os.remove(dest_path)
        raise


def _unzip_if_needed(zip_path: str, extract_dir: str) -> None:
    """Extract a zip file if it hasn't been extracted yet."""
    if not os.path.isfile(zip_path):
        return
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("Extracted %s to %s", zip_path, extract_dir)
    except Exception as e:
        logger.warning("Could not extract %s: %s", zip_path, e)


# ---------------------------------------------------------------------------
# Per-dataset download functions
# ---------------------------------------------------------------------------

def download_gadm_nigeria(dest_dir: str) -> str:
    """
    Download GADM 4.1 boundaries for Nigeria (GeoPackage format).

    Returns
    -------
    str
        Path to the downloaded .gpkg file.
    """
    dest_path = os.path.join(dest_dir, "gadm41_NGA.gpkg")
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_NGA.gpkg"
    _download_file(url, dest_path, "GADM 4.1 Nigeria boundaries")
    return dest_path


def download_worldpop_nigeria(dest_dir: str) -> str:
    """
    Download WorldPop constrained population raster for Nigeria (2020).

    Returns
    -------
    str
        Path to the downloaded .tif file.
    """
    dest_path = os.path.join(dest_dir, "nga_ppp_2020_constrained.tif")
    url = (
        "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/"
        "2020/maxar_v1/NGA/nga_ppp_2020_constrained.tif"
    )
    _download_file(url, dest_path, "WorldPop Nigeria 2020 population")
    return dest_path


def download_viirs_nightlights(dest_dir: str) -> str:
    """
    Download VIIRS annual nighttime lights composite for Nigeria.

    Note: The exact URL may vary by year. This downloads the 2021 annual
    average radiance composite clipped to Nigeria extent. If the URL is
    unavailable, users should manually place the nightlights raster at
    the expected path.

    Returns
    -------
    str
        Path to the expected nightlights .tif file.
    """
    dest_path = os.path.join(dest_dir, "nga_viirs_nightlights.tif")
    if os.path.isfile(dest_path):
        logger.info("Nightlights raster already exists: %s", dest_path)
        return dest_path

    logger.warning(
        "VIIRS nighttime lights raster not found at %s. "
        "Please download manually from https://eogdata.mines.edu/nighttime_light/ "
        "and place the Nigeria-clipped .tif at: %s",
        dest_path, dest_path,
    )
    return dest_path


def download_building_density(dest_dir: str) -> str:
    """
    Download GHSL building density raster for Nigeria.

    Note: GHSL-BUILT is a large global dataset. Users should download the
    relevant tile(s) covering Nigeria from:
    https://ghsl.jrc.ec.europa.eu/download.php?ds=bu

    Returns
    -------
    str
        Path to the expected building density .tif file.
    """
    dest_path = os.path.join(dest_dir, "nga_building_density.tif")
    if os.path.isfile(dest_path):
        logger.info("Building density raster already exists: %s", dest_path)
        return dest_path

    logger.warning(
        "Building density raster not found at %s. "
        "Please download from GHSL (https://ghsl.jrc.ec.europa.eu/download.php?ds=bu) "
        "and place the Nigeria-extent .tif at: %s",
        dest_path, dest_path,
    )
    return dest_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ensure_geospatial_data(cfg: dict) -> None:
    """
    Check for and download any missing geospatial data files.

    Parameters
    ----------
    cfg : dict
        Loaded config dictionary. Uses paths to determine what's needed
        and where to place downloads.
    """
    country_code = cfg.get("country", {}).get("code", "")
    if country_code != "NGA":
        return  # Only auto-download for Nigeria

    data_dir = os.path.dirname(cfg["paths"]["rwi_csv"])
    logger.info("Checking geospatial data for Nigeria in: %s", data_dir)

    # GADM boundaries
    gadm_path = cfg["paths"].get("gadm_gpkg", "")
    if gadm_path and not os.path.isfile(gadm_path):
        download_gadm_nigeria(data_dir)

    # WorldPop population
    pop_path = cfg["paths"].get("population_raster", "")
    if pop_path and not os.path.isfile(pop_path):
        download_worldpop_nigeria(data_dir)

    # Nighttime lights (manual download required)
    nl_path = cfg["paths"].get("nightlights_raster", "")
    if nl_path and not os.path.isfile(nl_path):
        download_viirs_nightlights(data_dir)

    # Building density (manual download required)
    bd_path = cfg["paths"].get("building_density_raster", "")
    if bd_path and not os.path.isfile(bd_path):
        download_building_density(data_dir)
