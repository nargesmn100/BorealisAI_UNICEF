"""
Data Pipeline Orchestrator

Runs Steps 01–05 in sequence to produce the clean modeling table.

Usage
-----
    python -m src.pipeline.run_pipeline
    # or within main.py via run_data_pipeline(cfg)

Steps:
    01  Build base grid from RWI CSV
    02  Sample proxy rasters onto grid (population, SMOD, accessibility)
    03  Assign GADM parishes + Urban/Rural/KMA subregions
    04  Prepare Jamaica poverty targets
    05  Merge features and targets into modeling table
"""

import logging
import os

import pandas as pd

from src.pipeline import step01_build_grid
from src.pipeline import step02_sample_proxies
from src.pipeline import step03_assign_admin
from src.pipeline import step04_prepare_targets
from src.pipeline import step05_merge_features
from src.utils.config_loader import load_config, setup_logging

logger = logging.getLogger(__name__)


def run_data_pipeline(
    cfg: dict,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """
    Run the full data pipeline from raw files to modeling table.

    Intermediate outputs are cached as Parquet files.  If a cached output
    exists and ``force_rerun=False``, that step is skipped.

    Parameters
    ----------
    cfg : dict
        Loaded config dictionary.
    force_rerun : bool
        If True, re-run all steps even if cached outputs exist.

    Returns
    -------
    pd.DataFrame
        Final modeling table.
    """
    logger.info("=" * 60)
    logger.info("Starting Jamaica Child Deprivation Data Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 01 — Base grid
    # ------------------------------------------------------------------
    grid_path = cfg["paths"]["grid_file"]
    if not force_rerun and os.path.isfile(grid_path):
        logger.info("[Step 01] Loading cached base grid from: %s", grid_path)
        grid = pd.read_parquet(grid_path)
    else:
        logger.info("[Step 01] Building base grid from RWI CSV...")
        grid = step01_build_grid.run(cfg)

    logger.info("[Step 01] Complete — %d grid cells.", len(grid))

    # ------------------------------------------------------------------
    # Step 02 — Sample proxies
    # ------------------------------------------------------------------
    proxies_path = cfg["paths"]["grid_with_proxies_file"]
    if not force_rerun and os.path.isfile(proxies_path):
        logger.info("[Step 02] Loading cached grid+proxies from: %s", proxies_path)
        grid_proxies = pd.read_parquet(proxies_path)
    else:
        logger.info("[Step 02] Sampling proxy rasters onto grid...")
        grid_proxies = step02_sample_proxies.run(cfg, grid)

    logger.info("[Step 02] Complete — %d rows, %d columns.", *grid_proxies.shape)

    # ------------------------------------------------------------------
    # Step 03 — Assign admin regions
    # ------------------------------------------------------------------
    admin_path = cfg["paths"]["grid_with_admin_file"]
    if not force_rerun and os.path.isfile(admin_path):
        logger.info("[Step 03] Loading cached grid+admin from: %s", admin_path)
        grid_admin = pd.read_parquet(admin_path)
    else:
        logger.info("[Step 03] Assigning administrative regions...")
        grid_admin = step03_assign_admin.run(cfg, grid_proxies)

    logger.info("[Step 03] Complete — %d rows.", len(grid_admin))

    # ------------------------------------------------------------------
    # Step 04 — Prepare targets
    # ------------------------------------------------------------------
    targets_path = cfg["paths"]["targets_file"]
    if not force_rerun and os.path.isfile(targets_path):
        logger.info("[Step 04] Loading cached targets from: %s", targets_path)
        targets = pd.read_csv(targets_path)
    else:
        logger.info("[Step 04] Preparing poverty targets...")
        targets = step04_prepare_targets.run(cfg)

    logger.info("[Step 04] Complete — %d target rows.", len(targets))

    # ------------------------------------------------------------------
    # Step 05 — Merge features + targets
    # ------------------------------------------------------------------
    table_path = cfg["paths"]["modeling_table_file"]
    if not force_rerun and os.path.isfile(table_path):
        logger.info("[Step 05] Loading cached modeling table from: %s", table_path)
        modeling_table = pd.read_parquet(table_path)
    else:
        logger.info("[Step 05] Merging features and targets...")
        modeling_table = step05_merge_features.run(cfg, grid_admin, targets)

    logger.info("[Step 05] Complete — modeling table: %d rows, %d columns.",
                *modeling_table.shape)

    logger.info("=" * 60)
    logger.info("Data Pipeline Complete")
    logger.info("Modeling sample size: %d / %d cells",
                modeling_table["in_modeling_sample"].sum(), len(modeling_table))
    logger.info("=" * 60)

    return modeling_table


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    run_data_pipeline(_cfg, force_rerun=False)
