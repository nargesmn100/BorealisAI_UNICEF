"""
Administrative Hierarchy Mappings
==================================

Shared lookup tables that map fine-grained admin units to coarser groupings.
Used by hierarchical validation, two-level CV, target computation, and the
modeling pipeline.

Adding a new country
---------------------
Add a ``<COUNTRY>_ADMIN_HIERARCHY`` dict mapping fine → coarse level names,
then register it in ``get_admin_hierarchy()`` via the country code.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nigeria: 37 states → 6 geopolitical zones
# ---------------------------------------------------------------------------

NIGERIA_GEOPOLITICAL_ZONES: dict[str, str] = {
    # South East
    "Abia": "South East",
    "Anambra": "South East",
    "Ebonyi": "South East",
    "Enugu": "South East",
    "Imo": "South East",
    # South South
    "Akwa Ibom": "South South",
    "Bayelsa": "South South",
    "Cross River": "South South",
    "Delta": "South South",
    "Edo": "South South",
    "Rivers": "South South",
    # South West
    "Ekiti": "South West",
    "Lagos": "South West",
    "Ogun": "South West",
    "Ondo": "South West",
    "Osun": "South West",
    "Oyo": "South West",
    # North Central
    "Benue": "North Central",
    "Kogi": "North Central",
    "Kwara": "North Central",
    "Nasarawa": "North Central",
    "Niger": "North Central",
    "Plateau": "North Central",
    "Federal Capital Territory": "North Central",
    # North East
    "Adamawa": "North East",
    "Bauchi": "North East",
    "Borno": "North East",
    "Gombe": "North East",
    "Taraba": "North East",
    "Yobe": "North East",
    # North West
    "Jigawa": "North West",
    "Kaduna": "North West",
    "Kano": "North West",
    "Katsina": "North West",
    "Kebbi": "North West",
    "Sokoto": "North West",
    "Zamfara": "North West",
}

# All 6 geopolitical zone names (ordered)
NIGERIA_ZONE_NAMES: list[str] = sorted(set(NIGERIA_GEOPOLITICAL_ZONES.values()))


# ---------------------------------------------------------------------------
# Generic accessor
# ---------------------------------------------------------------------------

def get_admin_hierarchy(country_code: str) -> dict[str, str] | None:
    """
    Return the fine→coarse admin mapping for a given country.

    Parameters
    ----------
    country_code : str
        ISO 3-letter country code (e.g. "NGA", "JAM").

    Returns
    -------
    dict or None
        Mapping {fine_unit_name: coarse_unit_name}, or None if not available.
    """
    if country_code.upper() == "NGA":
        return NIGERIA_GEOPOLITICAL_ZONES
    return None


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def add_geopolitical_zones(
    df: pd.DataFrame,
    zone_col: str = "subregion",
    output_col: str = "geopolitical_zone",
) -> pd.DataFrame:
    """
    Add a geopolitical zone column to a Nigeria modeling table.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``zone_col`` with state names (title case).
    zone_col : str
        Column containing state names.
    output_col : str
        Name of the new geopolitical zone column.

    Returns
    -------
    pd.DataFrame
        Copy with ``output_col`` added.
    """
    df = df.copy()
    df[output_col] = df[zone_col].map(NIGERIA_GEOPOLITICAL_ZONES).fillna("Unknown")
    n_mapped = (df[output_col] != "Unknown").sum()
    n_total = len(df)
    logger.info(
        "Geopolitical zone mapping: %d/%d cells mapped (%d unknown).",
        n_mapped, n_total, n_total - n_mapped,
    )
    return df


def add_state_urban_rural(
    df: pd.DataFrame,
    state_col: str = "subregion",
    urban_col: str = "is_urban",
    output_col: str = "state_urban_rural",
) -> pd.DataFrame:
    """
    Add a combined state×urban/rural identifier column.

    E.g. "Lagos" + is_urban=1  →  "Lagos_Urban"
         "Kano"  + is_urban=0  →  "Kano_Rural"

    Parameters
    ----------
    df : pd.DataFrame
    state_col : str
    urban_col : str
        Binary column (1=urban, 0=rural).
    output_col : str

    Returns
    -------
    pd.DataFrame
        Copy with ``output_col`` added.
    """
    df = df.copy()
    urban_label = df[urban_col].map({1: "Urban", 0: "Rural"}).fillna("Unknown")
    df[output_col] = df[state_col] + "_" + urban_label
    return df
