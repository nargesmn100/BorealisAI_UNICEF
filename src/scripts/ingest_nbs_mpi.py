"""
Ingest NBS Nigeria Multidimensional Poverty Index (MPI) Survey microdata.

Source: "Nigeria Multidimensional Poverty Index Survey/" folder (Stata .dta files)
        Downloaded from the National Bureau of Statistics (NBS).

The survey uses `a1` (state code 1–37) and `a2` (LGA code e.g. 101=ABA NORTH) as
geographic identifiers with the same state label mapping as MICS6.  Household
weights (`hh_wgt`) from Section A are applied throughout.

Dimensions ingested and aggregated to state level:
  Housing (Section J):
    nbs_floor_earth_pct    – % HH with earth/sand/dung floor (deprived)
    nbs_floor_finished_pct – % HH with finished floor (tiles/vinyl/marble/cement)

  Water & Sanitation (Section I):
    nbs_water_improved_pct  – % HH with piped/borehole/protected water source
    nbs_water_far_pct       – % HH needing ≥30 min to fetch water (round trip)
    nbs_toilet_improved_pct – % HH with non-shared improved toilet
    nbs_open_defecation_pct – % HH using bush/field/open (worst sanitation)

  Food Security (Section E — HFIAS 8-item scale):
    nbs_food_insecure_pct  – % HH with ≥3 food insecurity items (moderate+severe)
    nbs_food_severe_pct    – % HH with ≥5 food insecurity items (severe)

  Health access (Section F):
    nbs_health_far_pct      – % HH where nearest health facility takes >30 min

All outputs are on a 0–100 percentage scale.

Outputs
-------
Data/Nigeria/d1_external/nbs_mpi/nga_nbs_mpi_state.csv  – state-level features
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyreadstat

log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
MPI_DIR = ROOT / "Nigeria Multidimensional Poverty Index Survey"
OUT_DIR = ROOT / "Data/Nigeria/d1_external/nbs_mpi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── State label map — same as MICS6 ─────────────────────────────────────────
STATE_LABELS = {
    1: "Abia", 2: "Adamawa", 3: "Akwa Ibom", 4: "Anambra", 5: "Bauchi",
    6: "Bayelsa", 7: "Benue", 8: "Borno", 9: "Cross River", 10: "Delta",
    11: "Ebonyi", 12: "Edo", 13: "Ekiti", 14: "Enugu", 15: "Gombe",
    16: "Imo", 17: "Jigawa", 18: "Kaduna", 19: "Kano", 20: "Katsina",
    21: "Kebbi", 22: "Kogi", 23: "Kwara", 24: "Lagos", 25: "Nasarawa",
    26: "Niger", 27: "Ogun", 28: "Ondo", 29: "Osun", 30: "Oyo",
    31: "Plateau", 32: "Rivers", 33: "Sokoto", 34: "Taraba", 35: "Yobe",
    36: "Zamfara", 37: "Federal Capital Territory",
}

# Improved water source codes (i1)
IMPROVED_WATER_CODES = {1, 2, 3, 4, 5, 6}   # piped/borehole/protected spring/rainwater/sachet

# Improved (non-shared) toilet codes (i4)
IMPROVED_TOILET_CODES = {1, 2, 3, 5, 6}      # flush-to-sewer/septic/pit/ventilated-pit/slab


def _wload(path: Path, usecols: Optional[list] = None) -> pd.DataFrame:
    """Load a Stata DTA section file."""
    log.info("  Loading %s", path.name)
    kw = {"usecols": usecols} if usecols else {}
    df, _ = pyreadstat.read_dta(str(path), **kw)
    return df


def load_section_a() -> pd.DataFrame:
    """Load identification + weights (Section A)."""
    path = MPI_DIR / "SECTION A _ IDENTIFICATION.dta"
    df = _wload(path, usecols=["hh_id", "a1", "hh_wgt"])
    df["state"] = df["a1"].map(STATE_LABELS)
    df["hh_wgt"] = pd.to_numeric(df["hh_wgt"], errors="coerce").fillna(0)
    return df[["hh_id", "state", "hh_wgt"]]


def load_section_j(weights: pd.DataFrame) -> pd.DataFrame:
    """Section J: Housing characteristics."""
    path = MPI_DIR / "SECTION J_HOUSING CHARACTERISTICS_NEW.dta"
    df = _wload(path, usecols=["hh_id", "j1"])
    df = df.merge(weights, on="hh_id", how="left")
    df["j1"] = pd.to_numeric(df["j1"], errors="coerce")

    # Floor: 1=Earth/Sand, 2=Dung, 3=Wood planks, 4=Palm/Bamboo,
    #        5=Tiles, 6=Vinyl/asphalt, 7=Cement/concrete, 8=Marble/polished stone
    df["floor_earth"]    = df["j1"].isin([1, 2]).astype(float)
    df["floor_finished"] = df["j1"].isin([5, 6, 7, 8]).astype(float)
    return df[["hh_id", "state", "hh_wgt", "floor_earth", "floor_finished"]]


def load_section_i(weights: pd.DataFrame) -> pd.DataFrame:
    """Section I: Water and sanitation."""
    path = MPI_DIR / "SECTION I_ WATER AND SANITATION.dta"
    df = _wload(path, usecols=["hh_id", "i1", "i3", "i4", "i5"])
    df = df.merge(weights, on="hh_id", how="left")
    for col in ["i1", "i3", "i4", "i5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Water: improved = codes 1-6
    df["water_improved"] = df["i1"].isin(IMPROVED_WATER_CODES).astype(float)
    # Water: far = i3==2 (30 min or more round trip)
    df["water_far"]      = (df["i3"] == 2).astype(float)
    # Toilet: improved = codes 1-3,5,6; unshared = i5==2
    df["toilet_improved"] = (
        df["i4"].isin(IMPROVED_TOILET_CODES) & (df["i5"] == 2)
    ).astype(float)
    # Open defecation: code 7 in i4 (bush/field) or 8 (open), or NaN mapped to 0
    df["open_defecation"] = df["i4"].isin([7, 8]).astype(float)

    return df[["hh_id", "state", "hh_wgt",
               "water_improved", "water_far", "toilet_improved", "open_defecation"]]


def load_section_e(weights: pd.DataFrame) -> pd.DataFrame:
    """Section E: Food security (HFIAS 8-item)."""
    path = MPI_DIR / "SECTION E_FOOD SECURITY.dta"
    cols = ["hh_id"] + [f"e1{c}" for c in "abcdefgh"]
    df = _wload(path, usecols=cols)
    df = df.merge(weights, on="hh_id", how="left")
    # Each item: 1=Yes (deprived), 2=No → recode to 0/1
    item_cols = [f"e1{c}" for c in "abcdefgh"]
    for c in item_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").map({1.0: 1.0, 2.0: 0.0})

    df["insecure_count"] = df[item_cols].sum(axis=1)
    # Moderate+: ≥3 items; Severe: ≥5 items
    df["food_insecure"] = (df["insecure_count"] >= 3).astype(float)
    df["food_severe"]   = (df["insecure_count"] >= 5).astype(float)
    return df[["hh_id", "state", "hh_wgt", "food_insecure", "food_severe"]]


def load_section_f(weights: pd.DataFrame) -> pd.DataFrame:
    """Section F: Health facility access."""
    path = MPI_DIR / "SECTION F_HEALTH.dta"
    df = _wload(path, usecols=["hh_id", "f1"])
    df = df.merge(weights, on="hh_id", how="left")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    # f1: continuous minutes to nearest health facility
    # "Far" = >30 minutes (consistent with WHO standard)
    df["health_far"] = (df["f1"] > 30).astype(float)
    return df[["hh_id", "state", "hh_wgt", "health_far"]]


def _weighted_pct(df: pd.DataFrame, indicator: str, group: str = "state") -> pd.Series:
    """Compute weighted percentage of indicator by group."""
    def wpct(g):
        valid = g[indicator].notna() & g["hh_wgt"].notna() & (g["hh_wgt"] > 0)
        if valid.sum() == 0:
            return np.nan
        return np.average(g.loc[valid, indicator], weights=g.loc[valid, "hh_wgt"]) * 100

    return df.groupby(group).apply(wpct, include_groups=False).rename(f"nbs_{indicator}_pct")


def aggregate_to_state(
    weights: pd.DataFrame,
    j_df: pd.DataFrame,
    i_df: pd.DataFrame,
    e_df: pd.DataFrame,
    f_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all section aggregates into a single state-level table."""
    state_list = sorted(weights["state"].dropna().unique())
    state_df = pd.DataFrame({"subregion": state_list})

    # Helper: compute weighted pct series indexed by state name
    def compute(df, col):
        return _weighted_pct(df, col, group="state")

    state_df = state_df.merge(compute(j_df, "floor_earth").rename("nbs_floor_earth_pct").reset_index().rename(columns={"state": "subregion"}), on="subregion", how="left")
    state_df = state_df.merge(compute(j_df, "floor_finished").rename("nbs_floor_finished_pct").reset_index().rename(columns={"state": "subregion"}), on="subregion", how="left")
    state_df = state_df.merge(compute(i_df, "water_improved").rename("nbs_water_improved_pct").reset_index().rename(columns={"state": "subregion"}), on="subregion", how="left")
    state_df = state_df.merge(compute(i_df, "water_far").rename("nbs_water_far_pct").reset_index().rename(columns={"state": "subregion"}), on="subregion", how="left")
    state_df = state_df.merge(compute(i_df, "toilet_improved").rename("nbs_toilet_improved_pct").reset_index().rename(columns={"state": "subregion"}), on="subregion", how="left")
    state_df = state_df.merge(compute(i_df, "open_defecation").rename("nbs_open_defecation_pct").reset_index().rename(columns={"state": "subregion"}), on="subregion", how="left")
    state_df = state_df.merge(compute(e_df, "food_insecure").rename("nbs_food_insecure_pct").reset_index().rename(columns={"state": "subregion"}), on="subregion", how="left")
    state_df = state_df.merge(compute(e_df, "food_severe").rename("nbs_food_severe_pct").reset_index().rename(columns={"state": "subregion"}), on="subregion", how="left")
    state_df = state_df.merge(compute(f_df, "health_far").rename("nbs_health_far_pct").reset_index().rename(columns={"state": "subregion"}), on="subregion", how="left")

    return state_df


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if not MPI_DIR.exists():
        log.error("NBS MPI survey folder not found: %s", MPI_DIR)
        raise FileNotFoundError(MPI_DIR)

    log.info("Loading NBS MPI household weights (Section A) …")
    weights = load_section_a()
    log.info("  %d households, %d states", len(weights), weights["state"].nunique())

    log.info("Loading Section J (housing) …")
    j_df = load_section_j(weights)

    log.info("Loading Section I (water/sanitation) …")
    i_df = load_section_i(weights)

    log.info("Loading Section E (food security) …")
    e_df = load_section_e(weights)

    log.info("Loading Section F (health access) …")
    f_df = load_section_f(weights)

    log.info("Aggregating to state level …")
    state_df = aggregate_to_state(weights, j_df, i_df, e_df, f_df)

    out_path = OUT_DIR / "nga_nbs_mpi_state.csv"
    state_df.to_csv(out_path, index=False)
    log.info("Saved %d states → %s", len(state_df), out_path)

    # Print summary
    print("\n=== NBS MPI State-Level Features (sample) ===")
    feat_cols = [c for c in state_df.columns if c != "subregion"]
    print(f"{'State':<30}", "  ".join(f"{c[:20]:>20}" for c in feat_cols[:4]))
    for _, row in state_df.head(6).iterrows():
        vals = "  ".join(
            f"{row[c]:>20.1f}" if pd.notna(row.get(c)) else f"{'N/A':>20}"
            for c in feat_cols[:4]
        )
        print(f"{row['subregion']:<30} {vals}")
    print(f"\n  Total: {len(state_df)} states, {len(feat_cols)} features")


if __name__ == "__main__":
    main()
