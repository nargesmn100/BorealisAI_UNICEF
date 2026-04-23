"""
Process Nigeria DHS 2018 flat files (Kids Recode + Household Recode).

Reads fixed-width .DAT files using the accompanying .DCT Stata dictionary,
computes child deprivation indicators, and aggregates to cluster level.
Cluster-level output is keyed by (cluster_id, region, urban_rural) so it can
be spatially joined to GPS coordinates the moment those arrive.

Outputs
-------
Data/Nigeria/dhs/nga_dhs_cluster_deprivation.csv   – one row per cluster
Data/Nigeria/dhs/nga_dhs_state_deprivation.csv     – one row per state
"""

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
_DHS_RAW_DIR = ROOT / "Data" / "Nigeria" / "dhs" / "raw"


def _first_existing_dir(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


KR_DIR = _first_existing_dir([
    _DHS_RAW_DIR / "NGKR7BFL",
    ROOT / "NGKR7BFL",
])
HR_DIR = _first_existing_dir([
    _DHS_RAW_DIR / "NGHR7BFL",
    ROOT / "NGHR7BFL",
])
OUT_DIR = ROOT / "Data/Nigeria/dhs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DHS region code mapping (NGA DHS 2018, v024 = geopolitical zone)
# ---------------------------------------------------------------------------
REGION_MAP = {
    1: "North Central",
    2: "North East",
    3: "North West",
    4: "South East",
    5: "South South",
    6: "South West",
}


# ---------------------------------------------------------------------------
# DCT parser
# ---------------------------------------------------------------------------
def parse_dct(dct_path: Path) -> list[dict]:
    """Parse a Stata infix DCT file and return column specs."""
    pattern = re.compile(
        r"(?:str|byte|int|long|float|double)\s+(\S+)\s+\d+:\s*(\d+)-(\d+)"
    )
    cols = []
    with open(dct_path, "r", errors="replace") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                name = m.group(1)
                start = int(m.group(2)) - 1   # 0-based
                end = int(m.group(3))          # exclusive
                cols.append({"name": name, "start": start, "end": end})
    return cols


def read_dat(dat_path: Path, dct_path: Path, wanted: list[str]) -> pd.DataFrame:
    """Read a fixed-width DHS flat file, keeping only *wanted* columns."""
    all_cols = parse_dct(dct_path)
    keep = [c for c in all_cols if c["name"] in wanted]
    if not keep:
        raise ValueError(f"None of {wanted} found in {dct_path}")

    colspecs = [(c["start"], c["end"]) for c in keep]
    names    = [c["name"] for c in keep]

    log.info("Reading %s …", dat_path.name)
    df = pd.read_fwf(
        dat_path,
        colspecs=colspecs,
        names=names,
        header=None,
        dtype=str,
    )
    # convert to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    log.info("  %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------
_DHS_MISSING_ZSCORES = {9996, 9997, 9998, 9999, -9996, -9997, -9998, -9999}


def _zscore_flag(series: pd.Series, threshold: int = -200) -> pd.Series:
    """Return 1 if z-score < threshold, 0 if valid but ≥ threshold, NaN if missing."""
    missing = series.isin(_DHS_MISSING_ZSCORES)
    return pd.Series(
        np.where(missing, np.nan, (series < threshold).astype(float)),
        index=series.index,
    )


def _vaccination_complete(df: pd.DataFrame) -> pd.Series:
    """1 if child received BCG + DPT1 + measles, 0 otherwise, NaN if all missing."""
    # DHS codes: 1 = vaccination date on card, 2 = reported by mother, 8 = missing
    def received(col):
        return df[col].isin([1, 2]).astype(float).where(df[col] != 8, other=np.nan)

    bcg     = received("h2")
    dpt1    = received("h3")
    measles = received("h9")
    return (bcg * dpt1 * measles)   # NaN propagates automatically


IMPROVED_WATER = {
    11, 12, 13, 14,   # piped (dwelling / yard / public tap / neighbour)
    21,               # tube well / borehole
    31,               # protected dug well
    41,               # protected spring
    51,               # rainwater
    61,               # tanker-truck (often counted improved in NGA context)
    71, 72,           # bottled / sachet water
}

IMPROVED_SANITATION_EXCLUDE = {
    23,   # flush / pour-flush to open drain
    31,   # ventilated improved pit (VIP) – still acceptable but
    42,   # pit latrine without slab
    43,   # hanging latrine / bucket
    96,   # no facility / open defecation
}


def _improved_water(hv201: pd.Series) -> pd.Series:
    return hv201.isin(IMPROVED_WATER).astype(float).where(hv201.notna())


def _open_defecation(hv205: pd.Series) -> pd.Series:
    """1 = open defecation (hv205 == 31 open pit or 96 none)."""
    return hv205.isin({31, 96}).astype(float).where(hv205.notna())


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
def process_kids_recode() -> pd.DataFrame:
    """Compute child-level deprivation indicators and cluster summaries."""
    wanted = [
        "v001", "v002", "v005", "v024", "v025", "v190",
        "b4", "b5", "b8",          # sex, alive, age in years
        "hw1",                      # age in months
        "hw70", "hw71", "hw72",     # z-scores: wasting, stunting, underweight
        "h2", "h3", "h9",           # BCG, DPT1, measles
    ]
    df = read_dat(KR_DIR / "NGKR7BFL.DAT", KR_DIR / "NGKR7BFL.DCT", wanted)

    # Keep only living children under 5
    df = df[(df["b5"] == 1) & (df["b8"] < 5)].copy()
    log.info("Children under 5 (alive): %d", len(df))

    weight = df["v005"] / 1_000_000

    df["stunted"]       = _zscore_flag(df["hw71"], -200)
    df["wasted"]        = _zscore_flag(df["hw70"], -200)
    df["underweight"]   = _zscore_flag(df["hw72"], -200)
    df["vaccinated"]    = _vaccination_complete(df)

    # Cluster-level weighted means
    def wavg(grp, col):
        valid = grp[[col, "v005"]].dropna(subset=[col])
        if valid.empty:
            return np.nan
        w = valid["v005"] / 1_000_000
        return np.average(valid[col], weights=w)

    records = []
    for cluster_id, grp in df.groupby("v001"):
        records.append({
            "cluster_id":       cluster_id,
            "region_code":      grp["v024"].iloc[0],
            "urban_rural":      grp["v025"].iloc[0],
            "wealth_quintile":  grp["v190"].iloc[0],
            "n_children":       len(grp),
            "stunting_rate":    wavg(grp, "stunted"),
            "wasting_rate":     wavg(grp, "wasted"),
            "underweight_rate": wavg(grp, "underweight"),
            "vaccination_rate": wavg(grp, "vaccinated"),
        })

    cluster_df = pd.DataFrame(records)
    cluster_df["geopolitical_zone"] = cluster_df["region_code"].map(REGION_MAP)
    log.info("Clusters: %d", len(cluster_df))
    return cluster_df


def process_household_recode() -> pd.DataFrame:
    """Compute household-level WASH indicators and cluster summaries."""
    wanted = [
        "hv001", "hv002", "hv005", "hv024", "hv025",
        "hv201",   # drinking water source
        "hv205",   # toilet type
        "hv206",   # electricity
        "hv270",   # wealth index quintile
    ]
    df = read_dat(HR_DIR / "NGHR7BFL.DAT", HR_DIR / "NGHR7BFL.DCT", wanted)

    weight = df["hv005"] / 1_000_000

    df["improved_water"]   = _improved_water(df["hv201"])
    df["open_defecation"]  = _open_defecation(df["hv205"])
    df["has_electricity"]  = (df["hv206"] == 1).astype(float).where(df["hv206"].notna())

    def wavg(grp, col):
        valid = grp[[col, "hv005"]].dropna(subset=[col])
        if valid.empty:
            return np.nan
        w = valid["hv005"] / 1_000_000
        return np.average(valid[col], weights=w)

    records = []
    for cluster_id, grp in df.groupby("hv001"):
        records.append({
            "cluster_id":           cluster_id,
            "region_code":          grp["hv024"].iloc[0],
            "n_households":         len(grp),
            "improved_water_rate":  wavg(grp, "improved_water"),
            "open_defecation_rate": wavg(grp, "open_defecation"),
            "electricity_rate":     wavg(grp, "has_electricity"),
        })

    return pd.DataFrame(records)


def compute_composite_deprivation(kr: pd.DataFrame, hr: pd.DataFrame) -> pd.DataFrame:
    """Merge kids and household summaries; compute composite deprivation index."""
    merged = kr.merge(hr[["cluster_id", "improved_water_rate",
                            "open_defecation_rate", "electricity_rate"]],
                      on="cluster_id", how="left")

    # Simple composite: mean of available dimensions (higher = more deprived)
    deprivation_cols = [
        "stunting_rate", "wasting_rate", "open_defecation_rate",
    ]
    # Vaccination and water are *positive* — invert them
    merged["unvaccinated_rate"] = 1 - merged["vaccination_rate"]
    merged["unsafe_water_rate"] = 1 - merged["improved_water_rate"]

    all_dims = deprivation_cols + ["unvaccinated_rate", "unsafe_water_rate"]
    merged["deprivation_index"] = merged[all_dims].mean(axis=1)

    return merged


def aggregate_to_zone(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cluster-level indicators to geopolitical zone level."""
    cols = [
        "stunting_rate", "wasting_rate", "underweight_rate",
        "vaccination_rate", "improved_water_rate", "open_defecation_rate",
        "electricity_rate", "deprivation_index",
    ]
    existing = [c for c in cols if c in cluster_df.columns]
    zone_df = (
        cluster_df.groupby("geopolitical_zone")[existing]
        .mean()
        .reset_index()
        .rename(columns={"geopolitical_zone": "admin_name"})
    )
    return zone_df


def main():
    log.info("=== Processing Nigeria DHS 2018 ===")
    kr = process_kids_recode()
    hr = process_household_recode()
    cluster_df = compute_composite_deprivation(kr, hr)

    # Save cluster-level output (ready to join GPS when available)
    out_cluster = OUT_DIR / "nga_dhs_cluster_deprivation.csv"
    cluster_df.to_csv(out_cluster, index=False)
    log.info("Saved cluster-level: %s  (%d rows)", out_cluster, len(cluster_df))

    # Zone-level aggregation
    zone_df = aggregate_to_zone(cluster_df)
    out_zone = OUT_DIR / "nga_dhs_zone_deprivation.csv"
    zone_df.to_csv(out_zone, index=False)
    log.info("Saved zone-level: %s  (%d rows)", out_zone, len(zone_df))

    # Quick summary
    print("\n=== DHS Deprivation Summary (geopolitical zone means) ===")
    print(zone_df.to_string(index=False))
    print("\nTop zones by deprivation index:")
    if "deprivation_index" in zone_df.columns:
        print(zone_df.nlargest(6, "deprivation_index")[
            ["admin_name", "stunting_rate", "wasting_rate",
             "vaccination_rate", "deprivation_index"]
        ].to_string(index=False))
    print(f"\nNote: {len(cluster_df)} DHS clusters keyed by cluster_id.")
    print("With NGGE7BFL/NGGE7BFL.shp in Data/Nigeria/dhs/raw (or repo root), run:")
    print("  python -m src.scripts.merge_dhs_gps")
    print("  python -m src.scripts.validate_predictions_vs_dhs_gps")


if __name__ == "__main__":
    main()
