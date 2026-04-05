"""
Step 04 — Prepare Jamaica Poverty Targets

Extracts and standardises the poverty target data from ChPov_JAM_CUB.xlsx.

Data structure of ChPov_JAM_CUB.xlsx
--------------------------------------
- Country code: JAM, CUB
- Year: 2011, 2022
- Survey code: jam2011_imics, jam2022_imics
- SUBREGION_NAME: _T (national), Urban, Rural, Kingston Metropolitan Area (KMA)
- Dimension: _T, SEX, RESIDENCE, WEALTH_QUINTILE, etc.
- Subgroup: _T, F, M, R, U, Q1..Q5, etc.
- Key targets:
    - Prevalence, severe child poverty (%)
    - Prevalence, moderate child poverty (%)
    - Depth, severe/moderate child poverty (# deprivations)

Primary target selection
------------------------
We extract rows for:
  - country_code = "JAM"
  - survey_year  = 2022  (most recent)
  - Dimension    = "_T"  (no disaggregation)
  - Subgroup     = "_T"  (total)
  - SUBREGION_NAME: all four categories (national + 3 subnational)

This gives us one reference poverty value per geographic zone:
  - national
  - Urban
  - Rural
  - KMA

These values are used for:
  1. Admin reconciliation (Urban / Rural / KMA totals)
  2. Evaluation / baseline comparisons

Output saved to:  cfg["paths"]["targets_file"]   (CSV)

Columns in output:
  subregion            : one of {_T, Urban, Rural, KMA_label}
  severe_prevalence    : severe child poverty prevalence (%)
  moderate_prevalence  : moderate child poverty prevalence (%)
  severe_depth         : depth of severe poverty (# deprivations)
  moderate_depth       : depth of moderate poverty (# deprivations)
  survey_year          : year of survey
  survey_code          : survey identifier
  sample_size          : sample size from survey
"""

import logging
import os

import pandas as pd

from src.utils.config_loader import load_config, setup_logging

logger = logging.getLogger(__name__)

# Column name aliases for brevity
_COL_COUNTRY = "Country code"
_COL_YEAR = "Year"
_COL_SURVEY = "Survey code"
_COL_SUBREGION = "SUBREGION_NAME"
_COL_DIMENSION = "Dimension"
_COL_SUBGROUP = "Subgroup"
_COL_SAMPLE = "Sample"
_COL_SEV_PREV = "Prevalence, severe child poverty (%)"
_COL_MOD_PREV = "Prevalence, moderate child poverty (%)"
_COL_SEV_DEPTH = "Depth, severe child poverty (# deprivations)"
_COL_MOD_DEPTH = "Depth, moderate child poverty (# deprivations)"


def load_chpov_jamaica(cfg: dict) -> pd.DataFrame:
    """
    Load the ChPov_JAM_CUB.xlsx file and return all Jamaica rows.

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    pd.DataFrame
    """
    path = cfg["paths"]["chpov_jam_cub_xlsx"]
    logger.info("Loading ChPov_JAM_CUB.xlsx from: %s", path)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"ChPov_JAM_CUB.xlsx not found at: {path}")

    df = pd.read_excel(path, sheet_name="Sheet1")
    logger.info("Total rows loaded: %d, columns: %d", *df.shape)

    jam = df[df[_COL_COUNTRY] == cfg["targets"]["country_code"]].copy()
    logger.info("Jamaica rows: %d", len(jam))

    if len(jam) == 0:
        raise ValueError(
            f"No rows found for country code '{cfg['targets']['country_code']}' "
            f"in {path}.  Available codes: {sorted(df[_COL_COUNTRY].unique())}"
        )

    return jam


def extract_primary_targets(cfg: dict, jam_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to primary target rows: year=2022, total dimension/subgroup,
    across all subregions.

    Parameters
    ----------
    cfg : dict
    jam_df : pd.DataFrame
        All Jamaica rows from ChPov_JAM_CUB.xlsx.

    Returns
    -------
    pd.DataFrame
        One row per subregion with poverty target values.
    """
    year = cfg["targets"]["survey_year"]
    dim_total = cfg["targets"]["dimension_total"]
    sg_total = cfg["targets"]["subgroup_total"]

    mask = (
        (jam_df[_COL_YEAR] == year)
        & (jam_df[_COL_DIMENSION] == dim_total)
        & (jam_df[_COL_SUBGROUP] == sg_total)
    )
    filtered = jam_df[mask].copy()

    if len(filtered) == 0:
        # Fall back to most recent available year
        available_years = sorted(jam_df[_COL_YEAR].unique(), reverse=True)
        logger.warning(
            "No rows for year=%d with total dimension/subgroup. "
            "Available years: %s. Falling back to %d.",
            year, available_years, available_years[0],
        )
        year = available_years[0]
        mask = (
            (jam_df[_COL_YEAR] == year)
            & (jam_df[_COL_DIMENSION] == dim_total)
            & (jam_df[_COL_SUBGROUP] == sg_total)
        )
        filtered = jam_df[mask].copy()

    expected_subregions = {
        cfg["targets"]["subregion_national"],
        cfg["targets"]["subregion_urban"],
        cfg["targets"]["subregion_rural"],
        cfg["targets"]["subregion_kma"],
    }
    found_subregions = set(filtered[_COL_SUBREGION].tolist())
    missing_subregions = expected_subregions - found_subregions
    if missing_subregions:
        logger.warning(
            "Expected subregions not found in data: %s. "
            "Found: %s. Check data source.",
            missing_subregions, found_subregions,
        )

    logger.info(
        "Extracted %d primary target rows for year=%d:", len(filtered), year
    )
    for _, row in filtered.iterrows():
        logger.info(
            "  subregion=%-40s  severe=%.2f%%  moderate=%.2f%%",
            row[_COL_SUBREGION],
            row[_COL_SEV_PREV],
            row[_COL_MOD_PREV],
        )

    return filtered


def build_target_table(cfg: dict, filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the filtered poverty data into a clean target table.

    Parameters
    ----------
    cfg : dict
    filtered : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Clean target table with renamed columns.
    """
    kma_label = cfg["targets"]["subregion_kma"]

    targets = filtered[
        [
            _COL_SUBREGION,
            _COL_SEV_PREV,
            _COL_MOD_PREV,
            _COL_SEV_DEPTH,
            _COL_MOD_DEPTH,
            _COL_YEAR,
            _COL_SURVEY,
            _COL_SAMPLE,
        ]
    ].copy()

    targets = targets.rename(
        columns={
            _COL_SUBREGION: "subregion",
            _COL_SEV_PREV: "severe_prevalence",
            _COL_MOD_PREV: "moderate_prevalence",
            _COL_SEV_DEPTH: "severe_depth",
            _COL_MOD_DEPTH: "moderate_depth",
            _COL_YEAR: "survey_year",
            _COL_SURVEY: "survey_code",
            _COL_SAMPLE: "sample_size",
        }
    )

    # Standardise subregion labels to match Step 03 output
    # KMA label in data matches the config value exactly
    logger.info("ASSUMPTION: Subregion labels in ChPov match subregion column in grid:")
    logger.info("  'Urban' → is_urban=1, not in KMA parishes")
    logger.info("  'Rural' → is_urban=0")
    logger.info("  '%s' → is_urban=1, in KMA parishes %s", kma_label, cfg["geo"]["kma_parish_gids"])
    logger.info(
        "  '_T' (national) → used for validation, not for cell-level reconciliation"
    )

    targets = targets.reset_index(drop=True)
    return targets


def run(cfg: dict) -> pd.DataFrame:
    """
    Entry point for Step 04.

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    pd.DataFrame
        Clean target table.
    """
    jam_df = load_chpov_jamaica(cfg)
    filtered = extract_primary_targets(cfg, jam_df)
    targets = build_target_table(cfg, filtered)

    out_path = cfg["paths"]["targets_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    targets.to_csv(out_path, index=False)
    logger.info("Target table saved to: %s  (%d rows)", out_path, len(targets))

    return targets


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    run(_cfg)
