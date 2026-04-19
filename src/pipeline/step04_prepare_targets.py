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


def extract_quintile_targets(cfg: dict, jam_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Extract wealth quintile disaggregated targets from MICS data.

    Filters for Dimension="WEALTH_QUINTILE", Subgroup in Q1-Q5,
    for each subregion (Urban, Rural, KMA) and national.

    Parameters
    ----------
    cfg : dict
    jam_df : pd.DataFrame
        All Jamaica rows from ChPov_JAM_CUB.xlsx.

    Returns
    -------
    pd.DataFrame or None
        Quintile target table with columns: subregion, quintile, moderate_prevalence,
        severe_prevalence. Returns None if quintile data not found.
    """
    if not cfg["targets"].get("use_quintile_targets", False):
        logger.info("Quintile targets disabled in config. Skipping.")
        return None

    year = cfg["targets"]["survey_year"]
    quintile_dim = cfg["targets"].get("quintile_dimension", "WEALTH_QUINTILE")
    quintile_sgs = cfg["targets"].get("quintile_subgroups", ["Q1", "Q2", "Q3", "Q4", "Q5"])

    # Filter: WEALTH_QUINTILE dimension, Q1-Q5 subgroups, correct year
    mask = (
        (jam_df[_COL_YEAR] == year)
        & (jam_df[_COL_DIMENSION] == quintile_dim)
        & (jam_df[_COL_SUBGROUP].isin(quintile_sgs))
    )
    filtered = jam_df[mask].copy()

    if len(filtered) == 0:
        logger.warning("No quintile target rows found for year=%d, dimension=%s.", year, quintile_dim)
        return None

    # Build clean table
    rows = []
    for _, row in filtered.iterrows():
        rows.append({
            "subregion": row[_COL_SUBREGION],
            "quintile": row[_COL_SUBGROUP],
            "moderate_prevalence": row[_COL_MOD_PREV],
            "severe_prevalence": row[_COL_SEV_PREV],
        })

    qt = pd.DataFrame(rows)
    logger.info("Extracted %d quintile target rows:", len(qt))
    for _, row in qt.iterrows():
        logger.info(
            "  subregion=%-45s quintile=%s mod=%.1f%% sev=%.1f%%",
            row["subregion"], row["quintile"],
            row["moderate_prevalence"], row["severe_prevalence"],
        )

    # Save to interim
    output_prefix = cfg.get("country", {}).get("output_prefix", "jam")
    out_path = os.path.join(
        os.path.dirname(cfg["paths"]["targets_file"]),
        f"{output_prefix}_quintile_targets.csv",
    )
    qt.to_csv(out_path, index=False)
    logger.info("Quintile targets saved to: %s", out_path)

    return qt


def run(cfg: dict) -> pd.DataFrame:
    """
    Entry point for Step 04.

    Dispatches to the appropriate target extraction method based on
    cfg["targets"]["target_source"]:
      - "chpov_excel" (default, Jamaica) → parse ChPov_JAM_CUB.xlsx
      - "mics_microdata" (Nigeria etc.) → compute from MICS SPSS microdata

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    pd.DataFrame
        Clean target table.
    """
    target_source = cfg["targets"].get("target_source", "chpov_excel")

    if target_source == "mics_microdata":
        logger.info("Target source: MICS microdata (computing from SPSS files)...")
        from src.targets.compute_mics_deprivation import run_nigeria_targets
        targets = run_nigeria_targets(cfg)
        return targets

    # Default: Jamaica ChPov Excel extraction
    logger.info("Target source: ChPov Excel...")
    jam_df = load_chpov_jamaica(cfg)
    filtered = extract_primary_targets(cfg, jam_df)
    targets = build_target_table(cfg, filtered)

    out_path = cfg["paths"]["targets_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    targets.to_csv(out_path, index=False)
    logger.info("Target table saved to: %s  (%d rows)", out_path, len(targets))

    # Extract quintile targets if enabled
    extract_quintile_targets(cfg, jam_df)

    return targets


def extract_sex_disaggregated_targets(cfg: dict, jam_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract sex-disaggregated poverty targets (Male/Female) by zone.

    Filters ChPov_JAM_CUB.xlsx for Jamaica 2022, Dimension="SEX",
    Subgroup="F"/"M".

    Parameters
    ----------
    cfg : dict
    jam_df : pd.DataFrame
        All Jamaica rows from ChPov_JAM_CUB.xlsx.

    Returns
    -------
    pd.DataFrame
        Sex-disaggregated targets with columns: subregion, sex, severe_prevalence,
        moderate_prevalence, severe_depth, moderate_depth.
    """
    year = cfg["targets"]["survey_year"]
    dim_sex = cfg["targets"].get("dimension_sex", "SEX")
    subgroup_f = cfg["targets"].get("subgroup_female", "F")
    subgroup_m = cfg["targets"].get("subgroup_male", "M")

    mask = (
        (jam_df[_COL_YEAR] == year)
        & (jam_df[_COL_DIMENSION] == dim_sex)
        & (jam_df[_COL_SUBGROUP].isin([subgroup_f, subgroup_m]))
    )
    filtered = jam_df[mask].copy()

    if len(filtered) == 0:
        logger.warning(
            "No sex-disaggregated rows found for year=%d, dimension=%s. "
            "Available dimensions: %s",
            year, dim_sex, sorted(jam_df[_COL_DIMENSION].unique()),
        )
        return pd.DataFrame()

    targets = filtered[
        [_COL_SUBREGION, _COL_SUBGROUP, _COL_SEV_PREV, _COL_MOD_PREV,
         _COL_SEV_DEPTH, _COL_MOD_DEPTH]
    ].copy()

    targets = targets.rename(columns={
        _COL_SUBREGION: "subregion",
        _COL_SUBGROUP: "sex",
        _COL_SEV_PREV: "severe_prevalence",
        _COL_MOD_PREV: "moderate_prevalence",
        _COL_SEV_DEPTH: "severe_depth",
        _COL_MOD_DEPTH: "moderate_depth",
    })

    # Map subgroup codes to labels
    targets["sex"] = targets["sex"].map({subgroup_f: "Female", subgroup_m: "Male"}).fillna(targets["sex"])

    logger.info("Sex-disaggregated targets extracted: %d rows", len(targets))
    for _, row in targets.iterrows():
        logger.info(
            "  subregion=%-40s  sex=%-6s  moderate=%.2f%%  severe=%.2f%%",
            row["subregion"], row["sex"], row["moderate_prevalence"], row["severe_prevalence"],
        )

    sex_prefix = cfg.get("country", {}).get("output_prefix", "jam")
    out_path = os.path.join(os.path.dirname(cfg["paths"]["targets_file"]), f"{sex_prefix}_targets_sex.csv")
    targets.to_csv(out_path, index=False)
    logger.info("Sex-disaggregated targets saved to: %s", out_path)

    return targets


if __name__ == "__main__":
    _cfg = load_config()
    setup_logging(_cfg)
    run(_cfg)
