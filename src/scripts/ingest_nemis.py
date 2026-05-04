"""
Ingest Nigeria EMIS (Education Management Information System) school data
from NEMIS Excel workbooks.

Sources (Data/Nigeria/d1_external/nemis/):
  PRE-PRIMARY.xlsx  – pre-primary / ECCD schools
  PRIMARY.xlsx      – primary schools (P1–P6)
  JSS.xlsx          – Junior Secondary Schools (JSS1–JSS3)
  SSS.xlsx          – Senior Secondary Schools (SS1–SS3)

Each workbook has one row per school with columns: STATE, SCHOOL NAME, LGA,
SECTOR (Public/Private/IQS), LOCATION (Rural/Urban), and grade-level
enrolment by sex.

Aggregated to STATE level — features produced:
  nemis_pre_schools          – count of pre-primary schools
  nemis_primary_schools      – count of primary schools
  nemis_jss_schools          – count of JSS schools
  nemis_sss_schools          – count of SSS schools
  nemis_total_schools        – total schools (all levels)

  nemis_pre_enrol            – total pre-primary enrolment
  nemis_primary_enrol        – total primary enrolment (P1–P6)
  nemis_jss_enrol            – total JSS enrolment
  nemis_sss_enrol            – total SSS enrolment

  nemis_primary_pupil_per_school – avg pupils per primary school (class size proxy)
  nemis_jss_pupil_per_school     – avg pupils per JSS

  nemis_public_pct           – % schools that are public (primary + JSS)
  nemis_rural_pct            – % schools in rural areas (primary + JSS)

All counts refer to schools reporting ≥1 pupil (zero-enrolment schools excluded
from pupil/school ratios to avoid distorting the mean).

Outputs
-------
Data/Nigeria/d1_external/nemis/nga_nemis_state.csv  – state-level features
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
NEMIS_DIR = ROOT / "Data/Nigeria/d1_external/nemis"
OUT_DIR  = NEMIS_DIR  # same folder for the output CSV

# ── State name normalisation ──────────────────────────────────────────────────
# NEMIS uses ALL-CAPS names; GADM (our subregion) uses title-case.
_RAW_TO_GADM = {
    "ABUJA": "Federal Capital Territory",
    "FCT":   "Federal Capital Territory",
    "AKWA-IBOM": "Akwa Ibom",
    "AKWA IBOM": "Akwa Ibom",
    "BAYELSA": "Bayelsa",
    "CROSS RIVERS": "Cross River",
    "CROSS RIVER": "Cross River",
    "NASSARAWA": "Nasarawa",
    "NASARAWA":  "Nasarawa",
}

def _norm_state(raw: str) -> str:
    """Normalise NEMIS state string to GADM ADM1 title-case name."""
    s = str(raw).strip()
    upper = s.upper()
    if upper in _RAW_TO_GADM:
        return _RAW_TO_GADM[upper]
    return s.title()


def _load_sheet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        log.warning("File not found: %s — skipping", path)
        return None
    log.info("  Loading %s", path.name)
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    # Drop completely empty rows
    df = df.dropna(how="all")
    return df


def _total_enrol_cols(df: pd.DataFrame) -> list:
    """Return enrolment-total column names (T0TAL or TOTAL patterns)."""
    return [c for c in df.columns if re.search(r"T[0O]TAL", c, re.I)]


def _sum_enrol(df: pd.DataFrame) -> pd.Series:
    """Sum all enrolment-total columns per row."""
    cols = _total_enrol_cols(df)
    if not cols:
        return pd.Series(0, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)


def load_level(filename: str) -> Optional[pd.DataFrame]:
    """Load a single NEMIS level file, returning rows with state + enrolment."""
    df = _load_sheet(NEMIS_DIR / filename)
    if df is None:
        return None

    # Normalise STATE / LOCATION columns
    df["STATE"] = df["STATE"].astype(str).apply(_norm_state)
    if "LOCATION " in df.columns:
        df["LOCATION"] = df["LOCATION "].astype(str).str.strip().str.lower()
    elif "LOCATION" in df.columns:
        df["LOCATION"] = df["LOCATION"].astype(str).str.strip().str.lower()
    else:
        df["LOCATION"] = "unknown"

    if "SECTOR" in df.columns:
        df["SECTOR"] = df["SECTOR"].astype(str).str.strip().str.title()
    else:
        df["SECTOR"] = "Unknown"

    df["enrol_total"] = _sum_enrol(df)
    return df[["STATE", "SECTOR", "LOCATION", "enrol_total"]]


def aggregate_level(df: pd.DataFrame, level_prefix: str) -> pd.DataFrame:
    """Aggregate a single level's school data to state level."""
    cols_out = []
    records = []

    for state, grp in df.groupby("STATE"):
        active = grp[grp["enrol_total"] > 0]
        row = {
            "subregion":                          state,
            f"nemis_{level_prefix}_schools":      len(grp),
            f"nemis_{level_prefix}_enrol":        grp["enrol_total"].sum(),
        }
        if len(active) > 0:
            row[f"nemis_{level_prefix}_pupil_per_school"] = active["enrol_total"].mean()
        else:
            row[f"nemis_{level_prefix}_pupil_per_school"] = np.nan

        total = len(grp)
        row[f"nemis_{level_prefix}_public_pct"] = (
            (grp["SECTOR"] == "Public").sum() / total * 100 if total > 0 else np.nan
        )
        row[f"nemis_{level_prefix}_rural_pct"] = (
            (grp["LOCATION"] == "rural").sum() / total * 100 if total > 0 else np.nan
        )
        records.append(row)

    return pd.DataFrame(records)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    levels = {
        "pre":     "PRE-PRIMARY.xlsx",
        "primary": "PRIMARY.xlsx",
        "jss":     "JSS.xlsx",
        "sss":     "SSS.xlsx",
    }

    agg_dfs = []
    for prefix, fname in levels.items():
        df = load_level(fname)
        if df is None:
            log.warning("Skipping level %s — file missing", prefix)
            continue
        agg = aggregate_level(df, prefix)
        log.info("  %s: %d states", prefix, len(agg))
        agg_dfs.append(agg)

    if not agg_dfs:
        log.error("No NEMIS files loaded — aborting")
        return

    # Merge all levels on subregion
    merged = agg_dfs[0]
    for other in agg_dfs[1:]:
        merged = merged.merge(other, on="subregion", how="outer")

    # Derived composite columns
    school_count_cols = [c for c in merged.columns if c.endswith("_schools")]
    enrol_cols        = [c for c in merged.columns if c.endswith("_enrol") and "pct" not in c]

    merged["nemis_total_schools"] = merged[school_count_cols].fillna(0).sum(axis=1)
    merged["nemis_total_enrol"]   = merged[enrol_cols].fillna(0).sum(axis=1)

    # Overall public/rural pct (from primary + JSS where available)
    pub_cols  = [c for c in merged.columns if "public_pct" in c]
    rur_cols  = [c for c in merged.columns if "rural_pct"  in c]
    merged["nemis_public_pct"] = merged[pub_cols].mean(axis=1)
    merged["nemis_rural_pct"]  = merged[rur_cols].mean(axis=1)

    merged = merged.sort_values("subregion").reset_index(drop=True)

    out_path = OUT_DIR / "nga_nemis_state.csv"
    merged.to_csv(out_path, index=False)
    log.info("Saved %d states × %d features → %s", len(merged), len(merged.columns)-1, out_path)

    # Print summary
    print("\n=== NEMIS State-Level Features (sample) ===")
    show_cols = ["subregion", "nemis_primary_schools", "nemis_primary_enrol",
                 "nemis_primary_pupil_per_school", "nemis_public_pct", "nemis_rural_pct"]
    show_cols = [c for c in show_cols if c in merged.columns]
    print(merged[show_cols].head(8).to_string(index=False, float_format="{:.1f}".format))
    print(f"\n  Total: {len(merged)} states, {len(merged.columns)-1} feature columns")


if __name__ == "__main__":
    main()
