"""
Ingest Mo Ibrahim Index of African Governance (IIAG) 2024 edition.

Source: 2024-IIAG-scores.xlsx (project root)
Product: country-level governance scores for 54 African countries (2014–2023).

Since IIAG is national-level (not sub-national), the ingested scores are
broadcast as scalar constants to every cell in the modeling table.  They
provide institutional-context features — governance quality, social protection
coverage, education system capacity — that the pixel-level proxies (RWI,
nightlights, etc.) cannot capture on their own.

Key indicators extracted (all on 0–100 IIAG scale, higher = better):
  iiag_overall_governance   – GOVERNANCE (overall composite)
  iiag_security_rol         – SROL  (Security & Rule of Law)
  iiag_human_development    – HD    (Human Development composite)
  iiag_health               – HEALTH (Access, quality, child/maternal)
  iiag_education            – EDUC  (Enrolment, completion, quality)
  iiag_soc_protection       – SOCPROT (Social Protection & Welfare)
  iiag_abs_lived_poverty    – AbsLivPov (Absence of Lived Poverty)
  iiag_pov_reduction_pol    – PovRedPol (Poverty Reduction Policies)
  iiag_child_maternal_health – ContChildMatHealth

Outputs
-------
Data/Nigeria/d1_external/governance/nga_iiag_features.csv  – one row per year (2014-2023)
Data/Nigeria/d1_external/governance/nga_iiag_latest.csv   – latest year only (for joining)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
IIAG_XLSX = ROOT / "Data/Nigeria/d1_external/governance/2024-IIAG-scores.xlsx"
OUT_DIR   = ROOT / "Data/Nigeria/d1_external/governance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Indicators to extract: (output_col_name, IIAG series_id) ──────────────
INDICATORS = [
    ("iiag_overall_governance",    "GOVERNANCE"),
    ("iiag_security_rol",          "SROL"),
    ("iiag_human_development",     "HD"),
    ("iiag_health",                "HEALTH"),
    ("iiag_education",             "EDUC"),
    ("iiag_soc_protection",        "SOCPROT"),
    ("iiag_abs_lived_poverty",     "AbsLivPov"),
    ("iiag_pov_reduction_pol",     "PovRedPol"),
    ("iiag_child_maternal_health", "ContChildMatHealth"),
]


def load_iiag(path: Path = IIAG_XLSX) -> pd.DataFrame:
    """Parse IIAG workbook and return a tidy DataFrame with all years."""
    log.info("Loading IIAG from %s", path)
    raw = pd.read_excel(path, sheet_name=0, header=None)

    # Row 3 contains series IDs; row 6 is the data header (Country, Year, …)
    series_ids = raw.iloc[3, :].tolist()

    # Build column index map: series_id → column position
    sid_to_col = {}
    for col_idx, sid in enumerate(series_ids):
        if pd.notna(sid) and sid not in sid_to_col:
            sid_to_col[str(sid).strip()] = col_idx

    # Load data with row-6 as header
    df = pd.read_excel(path, sheet_name=0, header=6)
    df.columns = [str(c).strip() for c in df.columns]

    # The first two columns are Country and Year
    df = df.rename(columns={df.columns[0]: "Country", df.columns[1]: "Year"})
    df = df[df["Country"].notna()]

    # Filter Nigeria
    nga = df[df["Country"].str.strip().str.lower() == "nigeria"].copy()
    nga["Year"] = pd.to_numeric(nga["Year"], errors="coerce")
    nga = nga.dropna(subset=["Year"])
    nga["Year"] = nga["Year"].astype(int)

    if nga.empty:
        raise ValueError("No Nigeria rows found in IIAG workbook")

    log.info("  %d Nigeria rows (years %d–%d)", len(nga), nga["Year"].min(), nga["Year"].max())

    # Extract each target indicator
    col_names = list(df.columns)  # positional names like "Unnamed: 2" etc.
    records = []
    for _, row in nga.iterrows():
        rec = {"year": row["Year"]}
        for out_col, sid in INDICATORS:
            if sid in sid_to_col:
                raw_pos = sid_to_col[sid]
                # col_names[0]=Country, [1]=Year → data starts at pos 0
                # The excel column index maps directly to df column position
                if raw_pos < len(col_names):
                    val = row.iloc[raw_pos]
                    rec[out_col] = float(val) if pd.notna(val) else np.nan
                else:
                    rec[out_col] = np.nan
            else:
                log.warning("IIAG series ID %r not found in workbook", sid)
                rec[out_col] = np.nan
        records.append(rec)

    out = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if not IIAG_XLSX.exists():
        log.error("IIAG file not found: %s", IIAG_XLSX)
        raise FileNotFoundError(IIAG_XLSX)

    df = load_iiag(IIAG_XLSX)

    # Save all years
    all_path = OUT_DIR / "nga_iiag_features.csv"
    df.to_csv(all_path, index=False)
    log.info("Saved all years → %s", all_path)

    # Save latest year only
    latest = df[df["year"] == df["year"].max()].copy()
    latest_path = OUT_DIR / "nga_iiag_latest.csv"
    latest.to_csv(latest_path, index=False)
    log.info("Saved latest year (%d) → %s", latest["year"].iloc[0], latest_path)

    # Summary
    print("\n=== IIAG Nigeria — Latest Year ===")
    for _, row in latest.iterrows():
        print(f"  Year: {row['year']}")
        for out_col, _ in INDICATORS:
            val = row.get(out_col, np.nan)
            print(f"    {out_col:<35}  {val:.1f}" if pd.notna(val) else f"    {out_col:<35}  N/A")


if __name__ == "__main__":
    main()
