"""
Compute Multidimensional Child Deprivation Targets from MICS6 Microdata
========================================================================

Constructs UNICEF-standard multidimensional child deprivation indicators from
MICS6 SPSS household survey microdata.

6 dimensions of deprivation:
    1. Nutrition    — dietary diversity / breastfeeding proxy (Nigeria) or
                      anthropometric z-scores (Jamaica / generic)
    2. Health       — no immunization or recent illness without treatment
    3. Education    — not attending early childhood education
    4. WASH         — unsafe drinking water or sanitation
    5. Housing      — overcrowding or inadequate floor/roof/wall materials
    6. Information  — no access to radio, TV, telephone, or internet

A child is multidimensionally deprived if deprived in >= 2 dimensions (moderate)
or >= 3 dimensions (severe).

Output: CSV with columns matching the standard target schema:
    subregion, moderate_prevalence, severe_prevalence, moderate_depth,
    severe_depth, sample_size

Country-specific column mappings
---------------------------------
Nigeria MICS6 does NOT use the same column names as the generic MICS6 template.
Key differences:
  - No anthropometric z-scores (HAZ06/WAZ06/WHZ06) → use dietary diversity proxy
  - IM2/IM3 instead of IM011 for vaccination
  - CA1/CA5 instead of CA6 for illness treatment
  - UB6/UB7 instead of UB4/UB2 for education
  - HC7A/HC7B/HC9A/HC12/HC13 instead of HC7/HC8/HC9/HC10/HC15 for information

NUTRITION DIMENSION ASSUMPTION (DOCUMENT THIS FOR FUTURE CHANGES):
Nigeria MICS6 does NOT contain anthropometric z-scores (HAZ/WAZ/WHZ).
We use a tiered proxy strategy:
  1. Children 6-23 months with BD8 dietary data: Minimum Dietary Diversity (MDD).
     Deprived if consuming from <5 of 8 UNICEF food groups.
  2. Children 0-5 months without BD8 data: Deprived if never breastfed (BD2==2).
  3. Children 24-59 months without BD8 data: Conservatively marked NOT deprived.

If z-scores become available (e.g., from a different MICS round or anthropometry
module), replace the entire nutrition block with:
    HAZ06 < -200 | WAZ06 < -200 | WHZ06 < -200
The tiered proxy is a workaround for Nigeria MICS6's data limitations.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe column access helper
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return column if present, else a NaN Series matching the index."""
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


# ---------------------------------------------------------------------------
# Vectorized deprivation flag computation
# ---------------------------------------------------------------------------

def _detect_column_variant(df: pd.DataFrame) -> str:
    """
    Auto-detect whether the dataset uses Nigeria or generic MICS column names.

    Returns 'nigeria' or 'generic'.
    """
    # Nigeria markers: BD8C (dietary diversity), IM2 (vaccination), UB6 (education)
    nigeria_markers = ["BD8C", "IM2", "UB6"]
    generic_markers = ["HAZ06", "IM011", "UB4"]

    nga_hits = sum(1 for c in nigeria_markers if c in df.columns)
    gen_hits = sum(1 for c in generic_markers if c in df.columns)

    if nga_hits > gen_hits:
        return "nigeria"
    return "generic"


def _compute_nutrition_nigeria(merged: pd.DataFrame) -> pd.Series:
    """
    Nutrition deprivation for Nigeria MICS6 using tiered MDD proxy.

    Tier 1 — Children with BD8 dietary data (typically 6-23 months):
        Count food groups consumed from 8 UNICEF groups. Deprived if <5 groups.
    Tier 2 — Children without BD8 data and young (0-5 months proxy):
        Deprived if never breastfed (BD2 == 2).
    Tier 3 — All others: conservatively NOT deprived.
    """
    # 8 UNICEF food groups mapped to Nigeria MICS6 BD8 columns
    food_groups = [
        # 1. Grains, roots, tubers
        _safe_col(merged, "BD8C").eq(1) | _safe_col(merged, "BD8E").eq(1),
        # 2. Legumes and nuts
        _safe_col(merged, "BD8M").eq(1),
        # 3. Dairy (yogurt + cheese + animal milk)
        _safe_col(merged, "BD8A").eq(1) | _safe_col(merged, "BD8N").eq(1) | _safe_col(merged, "BD7E").eq(1),
        # 4. Flesh foods (organ meat + meat + fish)
        _safe_col(merged, "BD8I").eq(1) | _safe_col(merged, "BD8J").eq(1) | _safe_col(merged, "BD8L").eq(1),
        # 5. Eggs
        _safe_col(merged, "BD8K").eq(1),
        # 6. Vitamin-A rich fruits and vegetables
        _safe_col(merged, "BD8D").eq(1) | _safe_col(merged, "BD8F").eq(1) | _safe_col(merged, "BD8G").eq(1),
        # 7. Other fruits and vegetables
        _safe_col(merged, "BD8H").eq(1),
        # 8. Breastmilk
        _safe_col(merged, "BD3").eq(1),
    ]

    n_food_groups = sum(g.fillna(False).astype(int) for g in food_groups)

    # Tier 1: has dietary data → MDD threshold
    has_diet_data = _safe_col(merged, "BD8C").notna()
    dep_mdd = has_diet_data & (n_food_groups < 5)

    # Tier 2: no dietary data, never breastfed
    dep_bf = ~has_diet_data & (_safe_col(merged, "BD2").eq(2)).fillna(False)

    return (dep_mdd | dep_bf).astype(int)


def _compute_nutrition_generic(merged: pd.DataFrame) -> pd.Series:
    """Nutrition deprivation using anthropometric z-scores (generic MICS)."""
    dep = pd.Series(0, index=merged.index)
    for col in ["HAZ06", "WAZ06", "WHZ06"]:
        vals = _safe_col(merged, col)
        dep = dep | (vals.notna() & (vals < -200))  # MICS stores z-scores * 100
    return dep.astype(int)


def _compute_health_nigeria(merged: pd.DataFrame) -> pd.Series:
    """
    Health deprivation for Nigeria MICS6.

    Deprived if: no vaccination card (IM2 in {2,3}) OR
                 (had diarrhoea (CA1==1) AND did not seek treatment (CA5==2))
    """
    im2 = _safe_col(merged, "IM2")
    no_vacc = im2.isin([2, 3])

    ca1 = _safe_col(merged, "CA1")
    ca5 = _safe_col(merged, "CA5")
    untreated_illness = ca1.eq(1) & ca5.eq(2)

    return (no_vacc | untreated_illness).fillna(False).astype(int)


def _compute_health_generic(merged: pd.DataFrame) -> pd.Series:
    """Health deprivation using generic MICS columns."""
    im011 = _safe_col(merged, "IM011")
    no_vacc = im011.isin([0, 2, 3])

    ca6 = _safe_col(merged, "CA6")
    no_treatment = ca6.eq(2)

    return (no_vacc | no_treatment).fillna(False).astype(int)


def _compute_education_nigeria(merged: pd.DataFrame) -> pd.Series:
    """
    Education deprivation for Nigeria MICS6.

    Deprived if: never attended ECE (UB6==2) OR not currently attending (UB7==2)
    """
    ub6 = _safe_col(merged, "UB6")
    ub7 = _safe_col(merged, "UB7")
    return (ub6.eq(2) | ub7.eq(2)).fillna(False).astype(int)


def _compute_education_generic(merged: pd.DataFrame) -> pd.Series:
    """Education deprivation using generic MICS columns."""
    ub4 = _safe_col(merged, "UB4")
    ub2 = _safe_col(merged, "UB2")
    return (ub4.eq(2) | ub2.eq(2)).fillna(False).astype(int)


def _compute_wash(merged: pd.DataFrame) -> pd.Series:
    """WASH deprivation (same columns for Nigeria and generic)."""
    ws1 = _safe_col(merged, "WS1")
    ws11 = _safe_col(merged, "WS11")
    unsafe_water = ws1.ge(40) & ws1.notna()
    unsafe_sanitation = ws11.ge(30) & ws11.notna()
    return (unsafe_water | unsafe_sanitation).astype(int)


def _compute_housing(merged: pd.DataFrame) -> pd.Series:
    """Housing deprivation (same columns for Nigeria and generic)."""
    hc3 = _safe_col(merged, "HC3")
    hc4 = _safe_col(merged, "HC4")
    hc5 = _safe_col(merged, "HC5")
    bad_floor = hc3.isin([11, 12, 13])
    bad_roof = hc4.isin([11, 12, 13])
    bad_wall = hc5.isin([11, 12, 13])
    return (bad_floor | bad_roof | bad_wall).fillna(False).astype(int)


def _compute_information_nigeria(merged: pd.DataFrame) -> pd.Series:
    """
    Information deprivation for Nigeria MICS6.

    Nigeria uses HC7A (phone), HC7B (radio), HC9A (TV), HC12 (mobile),
    HC13 (internet) instead of HC7/HC8/HC9/HC10/HC15.
    """
    media_cols = ["HC7A", "HC7B", "HC9A", "HC12", "HC13"]
    has_any = pd.Series(False, index=merged.index)
    for col in media_cols:
        has_any = has_any | _safe_col(merged, col).eq(1)
    return (~has_any).astype(int)


def _compute_information_generic(merged: pd.DataFrame) -> pd.Series:
    """Information deprivation using generic MICS columns."""
    media_cols = ["HC7", "HC8", "HC9", "HC10", "HC15"]
    has_any = pd.Series(False, index=merged.index)
    for col in media_cols:
        has_any = has_any | _safe_col(merged, col).eq(1)
    return (~has_any).astype(int)


# ---------------------------------------------------------------------------
# Main computation entry point
# ---------------------------------------------------------------------------

def compute_child_deprivation_flags(
    ch_df: pd.DataFrame,
    hh_df: pd.DataFrame,
    fs_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute per-child binary deprivation flags across 6 dimensions (vectorized).

    Automatically detects whether the dataset uses Nigeria or generic MICS
    column names and dispatches to the appropriate dimension functions.

    Parameters
    ----------
    ch_df : pd.DataFrame
        Children's module (ch.sav)
    hh_df : pd.DataFrame
        Household module (hh.sav)
    fs_df : pd.DataFrame or None
        Food security module (fs.sav), optional — currently unused but
        reserved for future food security indicators.

    Returns
    -------
    pd.DataFrame
        One row per child with columns: HH1, HH2, HH7, chweight,
        dep_nutrition, dep_health, dep_education, dep_wash, dep_housing,
        dep_information, n_deprivations, moderate_deprived, severe_deprived
    """
    logger.info("Computing child deprivation flags for %d children...", len(ch_df))

    # Merge child + household on [HH1, HH2]
    hh_id_cols = ["HH1", "HH2"]
    if not all(c in ch_df.columns for c in hh_id_cols):
        raise ValueError(f"Child data missing join columns: {hh_id_cols}")
    if not all(c in hh_df.columns for c in hh_id_cols):
        raise ValueError(f"Household data missing join columns: {hh_id_cols}")

    merged = ch_df.merge(hh_df, on=hh_id_cols, how="left", suffixes=("", "_hh"))
    logger.info("Merged child+household: %d rows, %d columns", *merged.shape)

    # Detect column variant
    variant = _detect_column_variant(merged)
    logger.info("Detected MICS column variant: %s", variant)

    # Compute per-dimension flags
    if variant == "nigeria":
        dep_nutrition = _compute_nutrition_nigeria(merged)
        dep_health = _compute_health_nigeria(merged)
        dep_education = _compute_education_nigeria(merged)
        dep_information = _compute_information_nigeria(merged)
    else:
        dep_nutrition = _compute_nutrition_generic(merged)
        dep_health = _compute_health_generic(merged)
        dep_education = _compute_education_generic(merged)
        dep_information = _compute_information_generic(merged)

    # WASH and Housing use same columns in both variants
    dep_wash = _compute_wash(merged)
    dep_housing = _compute_housing(merged)

    # Build result DataFrame
    n_dep = (dep_nutrition + dep_health + dep_education +
             dep_wash + dep_housing + dep_information)

    result_df = pd.DataFrame({
        "HH1": merged["HH1"],
        "HH2": merged["HH2"],
        "HH7": merged["HH7"] if "HH7" in merged.columns else np.nan,
        "HH6": merged["HH6"] if "HH6" in merged.columns else np.nan,
        "chweight": merged["chweight"] if "chweight" in merged.columns else 1.0,
        "dep_nutrition": dep_nutrition.values,
        "dep_health": dep_health.values,
        "dep_education": dep_education.values,
        "dep_wash": dep_wash.values,
        "dep_housing": dep_housing.values,
        "dep_information": dep_information.values,
        "n_deprivations": n_dep.values,
        "moderate_deprived": (n_dep >= 2).astype(int).values,
        "severe_deprived": (n_dep >= 3).astype(int).values,
    })

    # Log per-dimension deprivation rates
    for dim in ["nutrition", "health", "education", "wash", "housing", "information"]:
        rate = result_df[f"dep_{dim}"].mean() * 100
        logger.info("  %s deprivation rate: %.1f%%", dim, rate)

    logger.info(
        "Deprivation flags computed. Moderate rate: %.1f%%, Severe rate: %.1f%%",
        result_df["moderate_deprived"].mean() * 100,
        result_df["severe_deprived"].mean() * 100,
    )
    return result_df


def aggregate_to_admin(
    child_df: pd.DataFrame,
    weight_col: str = "chweight",
    state_col: str = "HH7",
    state_labels: dict | None = None,
) -> pd.DataFrame:
    """
    Aggregate child-level deprivation to state-level weighted prevalence.

    State names are normalized to title case so they match GADM NAME_1
    (e.g., MICS "ABIA" → "Abia" to match GADM).

    Parameters
    ----------
    child_df : pd.DataFrame
        Per-child flags from compute_child_deprivation_flags()
    weight_col : str
        Survey weight column
    state_col : str
        State/admin column
    state_labels : dict or None
        Mapping from state code to state name (from SPSS value labels)

    Returns
    -------
    pd.DataFrame
        State-level targets with columns matching standard schema.
    """
    logger.info("Aggregating deprivation to admin level using '%s' weights...", weight_col)

    # Clean weights
    child_df = child_df.copy()
    child_df[weight_col] = pd.to_numeric(child_df[weight_col], errors="coerce").fillna(0)

    rows = []
    for state_code, group in child_df.groupby(state_col):
        w = group[weight_col].values
        total_weight = w.sum()

        if total_weight == 0:
            logger.warning("State %s has zero total weight, skipping.", state_code)
            continue

        # Weighted prevalence
        mod_prev = np.average(group["moderate_deprived"].values, weights=w) * 100
        sev_prev = np.average(group["severe_deprived"].values, weights=w) * 100

        # Weighted depth (mean number of deprivations among deprived)
        mod_mask = group["moderate_deprived"] == 1
        sev_mask = group["severe_deprived"] == 1

        if mod_mask.sum() > 0:
            mod_depth = np.average(
                group.loc[mod_mask, "n_deprivations"].values,
                weights=group.loc[mod_mask, weight_col].values,
            )
        else:
            mod_depth = 0.0

        if sev_mask.sum() > 0:
            sev_depth = np.average(
                group.loc[sev_mask, "n_deprivations"].values,
                weights=group.loc[sev_mask, weight_col].values,
            )
        else:
            sev_depth = 0.0

        # Map state code to name
        if state_labels and state_code in state_labels:
            subregion = state_labels[state_code]
        else:
            subregion = str(int(state_code)) if pd.notna(state_code) else str(state_code)

        # Normalize to title case so MICS "ABIA" matches GADM "Abia"
        subregion = subregion.strip().title()

        rows.append({
            "subregion": subregion,
            "moderate_prevalence": round(mod_prev, 2),
            "severe_prevalence": round(sev_prev, 2),
            "moderate_depth": round(mod_depth, 2),
            "severe_depth": round(sev_depth, 2),
            "sample_size": len(group),
        })

    targets = pd.DataFrame(rows)
    logger.info("State-level targets computed for %d states.", len(targets))
    logger.info(
        "Moderate prevalence range: [%.1f%%, %.1f%%]",
        targets["moderate_prevalence"].min(),
        targets["moderate_prevalence"].max(),
    )
    logger.info(
        "Severe prevalence range: [%.1f%%, %.1f%%]",
        targets["severe_prevalence"].min(),
        targets["severe_prevalence"].max(),
    )
    return targets


def _weighted_aggregate(group: pd.DataFrame, weight_col: str) -> dict:
    """
    Compute weighted prevalence and depth for a group of children.

    Returns a dict with keys: moderate_prevalence, severe_prevalence,
    moderate_depth, severe_depth, sample_size.
    """
    w = group[weight_col].values.astype(float)
    total_weight = w.sum()
    if total_weight == 0:
        return None

    mod_prev = np.average(group["moderate_deprived"].values, weights=w) * 100
    sev_prev = np.average(group["severe_deprived"].values, weights=w) * 100

    mod_mask = group["moderate_deprived"] == 1
    sev_mask = group["severe_deprived"] == 1

    mod_depth = (
        np.average(group.loc[mod_mask, "n_deprivations"].values,
                   weights=group.loc[mod_mask, weight_col].values)
        if mod_mask.sum() > 0 else 0.0
    )
    sev_depth = (
        np.average(group.loc[sev_mask, "n_deprivations"].values,
                   weights=group.loc[sev_mask, weight_col].values)
        if sev_mask.sum() > 0 else 0.0
    )

    return {
        "moderate_prevalence": round(mod_prev, 2),
        "severe_prevalence": round(sev_prev, 2),
        "moderate_depth": round(mod_depth, 2),
        "severe_depth": round(sev_depth, 2),
        "sample_size": len(group),
    }


def compute_multilevel_targets(
    child_df: pd.DataFrame,
    weight_col: str = "chweight",
    state_col: str = "HH7",
    urban_col: str = "HH6",
    state_labels: dict | None = None,
    geopolitical_zones: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute deprivation targets at 4 hierarchical levels from child-level flags.

    Levels produced:
        "national"          — 1 row  (all Nigeria)
        "geopolitical_zone" — 6 rows (NW, NE, NC, SW, SS, SE)
        "state"             — 37 rows (one per state)
        "state_urban_rural" — up to 74 rows (state × urban/rural)

    Parameters
    ----------
    child_df : pd.DataFrame
        Per-child flags from compute_child_deprivation_flags(), must include
        HH7 (state code) and HH6 (urban/rural: 1=urban, 2=rural).
    weight_col : str
        Survey weight column.
    state_col : str
        State identifier column (integer codes from SPSS).
    urban_col : str
        Urban/rural column (1=urban, 2=rural in MICS coding).
    state_labels : dict or None
        Mapping {state_code: state_name} from SPSS value labels.
    geopolitical_zones : dict or None
        Mapping {state_name: zone_name}. Defaults to Nigeria zones if None.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "national", "geopolitical_zone", "state", "state_urban_rural".
        Each DataFrame has schema:
            group_id, moderate_prevalence, severe_prevalence,
            moderate_depth, severe_depth, sample_size
    """
    from src.utils.admin_mappings import NIGERIA_GEOPOLITICAL_ZONES

    if geopolitical_zones is None:
        geopolitical_zones = NIGERIA_GEOPOLITICAL_ZONES

    df = child_df.copy()
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)

    # Resolve state names
    def _state_name(code) -> str:
        if state_labels and code in state_labels:
            return state_labels[code].strip().title()
        try:
            return str(int(code))
        except (ValueError, TypeError):
            return str(code)

    df["_state_name"] = df[state_col].map(_state_name)

    # Urban/rural label (MICS: HH6==1 urban, HH6==2 rural)
    if urban_col in df.columns:
        df["_urban_label"] = df[urban_col].map({1: "Urban", 2: "Rural"}).fillna("Unknown")
    else:
        logger.warning("Urban/rural column '%s' not found. state_urban_rural level will be empty.", urban_col)
        df["_urban_label"] = "Unknown"

    # Geopolitical zone
    df["_geo_zone"] = df["_state_name"].map(geopolitical_zones).fillna("Unknown")

    results: dict[str, pd.DataFrame] = {}

    # --- Level 0: National ---
    nat = _weighted_aggregate(df, weight_col)
    if nat:
        results["national"] = pd.DataFrame([{"group_id": "National", **nat}])
    logger.info("National target: moderate=%.1f%%, severe=%.1f%%",
                nat["moderate_prevalence"] if nat else 0,
                nat["severe_prevalence"] if nat else 0)

    # --- Level 1: Geopolitical zones ---
    geo_rows = []
    for zone, grp in df.groupby("_geo_zone"):
        if zone == "Unknown":
            continue
        agg = _weighted_aggregate(grp, weight_col)
        if agg:
            geo_rows.append({"group_id": zone, **agg})
    results["geopolitical_zone"] = pd.DataFrame(geo_rows).sort_values("group_id").reset_index(drop=True)
    logger.info("Geopolitical zone targets: %d zones", len(results["geopolitical_zone"]))

    # --- Level 2: States ---
    state_rows = []
    for state, grp in df.groupby("_state_name"):
        agg = _weighted_aggregate(grp, weight_col)
        if agg:
            state_rows.append({"group_id": state, **agg})
    results["state"] = pd.DataFrame(state_rows).sort_values("group_id").reset_index(drop=True)
    logger.info("State targets: %d states. Moderate range: [%.1f%%, %.1f%%]",
                len(results["state"]),
                results["state"]["moderate_prevalence"].min(),
                results["state"]["moderate_prevalence"].max())

    # --- Level 3: State × Urban/Rural ---
    sur_rows = []
    for (state, urban), grp in df.groupby(["_state_name", "_urban_label"]):
        if urban == "Unknown":
            continue
        agg = _weighted_aggregate(grp, weight_col)
        if agg:
            sur_rows.append({"group_id": f"{state}_{urban}", **agg})
    sur_df = pd.DataFrame(sur_rows)
    results["state_urban_rural"] = (
        sur_df.sort_values("group_id").reset_index(drop=True)
        if not sur_df.empty else sur_df
    )
    logger.info("State×Urban/Rural targets: %d groups", len(results["state_urban_rural"]))

    return results


def run_nigeria_targets(cfg: dict) -> pd.DataFrame:
    """
    Entry point: compute Nigeria MICS6 deprivation targets.

    Reads SPSS files (ch.sav, hh.sav, optionally fs.sav), computes per-child
    deprivation flags, and aggregates to state-level targets.

    Parameters
    ----------
    cfg : dict
        Config dictionary with paths.mics_data_dir pointing to SPSS files.

    Returns
    -------
    pd.DataFrame
        State-level targets in standard schema.
    """
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "pyreadstat is required for MICS SPSS data. "
            "Install with: pip install pyreadstat"
        )

    mics_dir = cfg["paths"]["mics_data_dir"]
    logger.info("Loading MICS6 SPSS data from: %s", mics_dir)

    # Load children's module
    ch_path = os.path.join(mics_dir, "ch.sav")
    logger.info("Loading ch.sav...")
    ch_df, ch_meta = pyreadstat.read_sav(ch_path)
    logger.info("  Children module: %d rows, %d columns", *ch_df.shape)

    # Load household module
    hh_path = os.path.join(mics_dir, "hh.sav")
    logger.info("Loading hh.sav...")
    hh_df, hh_meta = pyreadstat.read_sav(hh_path)
    logger.info("  Household module: %d rows, %d columns", *hh_df.shape)

    # Load food security module (optional)
    fs_df = None
    fs_path = os.path.join(mics_dir, "fs.sav")
    if os.path.isfile(fs_path):
        logger.info("Loading fs.sav...")
        fs_df, _ = pyreadstat.read_sav(fs_path)
        logger.info("  Food security module: %d rows, %d columns", *fs_df.shape)

    # Extract state code → state name mapping from SPSS value labels
    state_labels = {}
    if "HH7" in ch_meta.variable_value_labels:
        state_labels = {
            int(k) if isinstance(k, float) else k: v
            for k, v in ch_meta.variable_value_labels["HH7"].items()
        }
        logger.info("State labels from SPSS metadata: %d states", len(state_labels))
    else:
        logger.warning(
            "No value labels found for HH7 in ch.sav metadata. "
            "State codes will be used as-is."
        )

    # Log available columns for debugging
    logger.info("Children columns (first 30): %s", list(ch_df.columns[:30]))
    logger.info("Household columns (first 30): %s", list(hh_df.columns[:30]))

    # Compute per-child deprivation flags
    child_flags = compute_child_deprivation_flags(ch_df, hh_df, fs_df)

    # Aggregate to state level (primary targets file — used by pipeline for reconciliation)
    targets = aggregate_to_admin(
        child_flags,
        weight_col="chweight",
        state_col="HH7",
        state_labels=state_labels,
    )

    out_path = cfg["paths"]["targets_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    targets.to_csv(out_path, index=False)
    logger.info("Nigeria state targets saved to: %s (%d rows)", out_path, len(targets))

    # Compute multilevel targets (national / geo-zone / state / state×urban-rural)
    # These are used by hierarchical validation to test cross-level generalization.
    try:
        multilevel = compute_multilevel_targets(
            child_flags,
            weight_col="chweight",
            state_col="HH7",
            urban_col="HH6",
            state_labels=state_labels,
        )
        interim_dir = cfg["paths"]["interim_dir"]
        os.makedirs(interim_dir, exist_ok=True)
        for level_name, level_df in multilevel.items():
            level_path = os.path.join(interim_dir, f"nga_targets_{level_name}.csv")
            level_df.to_csv(level_path, index=False)
            logger.info("Multilevel targets [%s] saved to: %s (%d rows)",
                        level_name, level_path, len(level_df))
    except Exception as e:
        logger.warning("Could not compute multilevel targets: %s", e)

    return targets
