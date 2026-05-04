"""
DHS 2018 Nigeria — HAZ-based Nutrition Targets
===============================================

Parses NGKR7BFL.DAT (Children's Recode, Nigeria DHS 2018) to compute
state-level stunting prevalence from height-for-age z-scores (hw70, WHO
standard), then aggregates to GADM state names for use as the nutrition
dimension training target.

Variable mapping (confirmed from NGKR7BFL.DCT / .DO):
  v001   — cluster number  (links to NGGE7BFL.shp DHSCLUST → state)
  v005   — sample weight   (DHS weight / 1,000,000 = decimal weight)
  hw1    — child age in months at time of measurement
  hw70   — Height/Age standard deviation × 100 (WHO new ref)
             valid range −600 to +600; ≥9996 = flagged/missing
  b5     — child alive (1=yes, 0=dead)

Stunting thresholds (WHO):
  Moderate (moderate or severe): HAZ < −2.0  →  hw70 < −200
  Severe:                        HAZ < −3.0  →  hw70 < −300

State assignment strategy:
  DHS KR v024 gives only 6 geopolitical zones, not states.
  We link each cluster (v001) to DHSCLUST in the GPS shapefile
  (NGGE7BFL.shp), which carries ADM1NAME (state name in UPPER CASE).
  Clusters with lat/lon = 0 (suppressed for privacy) are excluded.

Survey design note:
  DHS 2018 is from 2018; MICS 2021 is from 2021.  The state-level
  stunting rates are reasonably stable (national stunting declined
  ~2 pp over that period), but users should be aware of the 3-year
  gap.  The DHS 2018 HAZ targets are used as a PROXY replacement for
  the MICS MDD target — scientifically stronger than MDD, with minor
  temporal uncertainty.

Outputs:
  Data/interim/nga/nga_dhs_haz_targets.csv
      subregion, haz_moderate_prev, haz_severe_prev, haz_n
"""

import logging
import os
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State name harmonisation: GPS ADM1NAME (UPPER CASE) → GADM title case
# ---------------------------------------------------------------------------

_GPS_TO_GADM: dict[str, str] = {
    "FCT ABUJA":   "Federal Capital Territory",
    "ABIA":        "Abia",
    "ADAMAWA":     "Adamawa",
    "AKWA IBOM":   "Akwa Ibom",
    "ANAMBRA":     "Anambra",
    "BAUCHI":      "Bauchi",
    "BAYELSA":     "Bayelsa",
    "BENUE":       "Benue",
    "BORNO":       "Borno",
    "CROSS RIVER": "Cross River",
    "DELTA":       "Delta",
    "EBONYI":      "Ebonyi",
    "EDO":         "Edo",
    "EKITI":       "Ekiti",
    "ENUGU":       "Enugu",
    "GOMBE":       "Gombe",
    "IMO":         "Imo",
    "JIGAWA":      "Jigawa",
    "KADUNA":      "Kaduna",
    "KANO":        "Kano",
    "KATSINA":     "Katsina",
    "KEBBI":       "Kebbi",
    "KOGI":        "Kogi",
    "KWARA":       "Kwara",
    "LAGOS":       "Lagos",
    "NASARAWA":    "Nasarawa",
    "NIGER":       "Niger",
    "OGUN":        "Ogun",
    "ONDO":        "Ondo",
    "OSUN":        "Osun",
    "OYO":         "Oyo",
    "PLATEAU":     "Plateau",
    "RIVERS":      "Rivers",
    "SOKOTO":      "Sokoto",
    "TARABA":      "Taraba",
    "YOBE":        "Yobe",
    "ZAMFARA":     "Zamfara",
}


def _harmonise_state(raw: str) -> str:
    key = raw.strip().upper()
    return _GPS_TO_GADM.get(key, raw.strip().title())


# ---------------------------------------------------------------------------
# Parse NGKR7BFL.DAT using the .DCT fixed-width layout
# ---------------------------------------------------------------------------

def _parse_dct(dct_path: str) -> dict[str, tuple[int, int]]:
    """Return {varname: (start_0idx, end_0idx)} from a DHS CSPro .DCT file."""
    colspecs: dict[str, tuple[int, int]] = {}
    with open(dct_path) as f:
        for line in f:
            m = re.match(r'\s+\w+\s+(\w+)\s+1:\s*(\d+)-\s*(\d+)', line)
            if m:
                name, start, end = m.group(1), int(m.group(2)), int(m.group(3))
                colspecs[name] = (start - 1, end)  # convert to 0-indexed
    return colspecs


def load_kr_haz(kr_dir: str) -> pd.DataFrame:
    """
    Load NGKR7BFL.DAT and return a DataFrame with:
        v001   cluster number
        v005   survey weight (raw × 1e6)
        hw1    child age in months
        hw70   HAZ × 100  (−600 to 600 = valid; ≥9996 = missing)
        b5     child alive (1 = yes)
    """
    dat_path = os.path.join(kr_dir, "NGKR7BFL.DAT")
    dct_path = os.path.join(kr_dir, "NGKR7BFL.DCT")

    for p in [dat_path, dct_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    logger.info("Parsing NGKR7BFL.DCT for column positions…")
    colspecs = _parse_dct(dct_path)

    needed = ["v001", "v005", "hw1", "hw70", "b5"]
    missing = [v for v in needed if v not in colspecs]
    if missing:
        raise KeyError(f"Variables not found in DCT: {missing}")

    avail = {k: colspecs[k] for k in needed}
    logger.info("Reading NGKR7BFL.DAT (fixed-width)…")
    df = pd.read_fwf(
        dat_path,
        colspecs=list(avail.values()),
        names=list(avail.keys()),
        header=None,
        dtype=str,
        na_values=[""],
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    logger.info("  KR rows loaded: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Compute state-level stunting prevalence
# ---------------------------------------------------------------------------

def compute_haz_state_targets(
    kr_dir: str,
    gps_shp: str,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        subregion            — GADM state name
        haz_moderate_prev    — % stunted (HAZ < −2) among under-5s measured
        haz_severe_prev      — % severely stunted (HAZ < −3)
        haz_n                — number of children with valid HAZ measurement
    """
    # 1. Load KR
    kr = load_kr_haz(kr_dir)

    # 2. Filter: alive children (b5 == 1), under 60 months (hw1 < 60)
    kr = kr[kr["b5"].eq(1)].copy()
    kr = kr[kr["hw1"].lt(60) & kr["hw1"].ge(0)].copy()
    logger.info("  After alive + age filter: %d children", len(kr))

    # 3. Valid HAZ: hw70 in (−600, 600) — DHS flags ≥9996
    hw70_raw = kr["hw70"].copy()
    valid_mask = (hw70_raw > -600) & (hw70_raw < 600)
    kr["haz"] = hw70_raw.where(valid_mask, np.nan)
    logger.info(
        "  Valid HAZ measurements: %d / %d (%.1f%%)",
        valid_mask.sum(), len(kr), valid_mask.mean() * 100,
    )

    # 4. Link cluster → state via GPS shapefile
    gps = gpd.read_file(gps_shp)[["DHSCLUST", "ADM1NAME", "LATNUM", "LONGNUM"]].copy()
    # Drop clusters with suppressed coordinates (lat==0, lon==0)
    gps = gps[(gps["LATNUM"] != 0) | (gps["LONGNUM"] != 0)].copy()
    gps["subregion"] = gps["ADM1NAME"].map(_harmonise_state)
    cluster_state = gps.set_index("DHSCLUST")["subregion"].to_dict()

    kr["subregion"] = kr["v001"].map(cluster_state)
    unmatched = kr["subregion"].isna().sum()
    if unmatched:
        logger.warning("  %d KR rows could not be matched to a state cluster", unmatched)
    kr = kr.dropna(subset=["subregion"])

    # 5. Aggregate to state level (weighted)
    rows = []
    for state, grp in kr.groupby("subregion"):
        w = (grp["v005"] / 1_000_000).fillna(0).values
        haz = grp["haz"].values
        valid = ~np.isnan(haz)
        if not valid.any() or w[valid].sum() == 0:
            logger.warning("  No valid HAZ data for state: %s", state)
            continue
        w_valid = w[valid]
        h_valid = haz[valid]
        mod_prev = np.average(h_valid < -200, weights=w_valid) * 100
        sev_prev = np.average(h_valid < -300, weights=w_valid) * 100
        rows.append({
            "subregion":         state,
            "haz_moderate_prev": round(mod_prev, 2),
            "haz_severe_prev":   round(sev_prev, 2),
            "haz_n":             int(valid.sum()),
        })

    targets = pd.DataFrame(rows).sort_values("subregion").reset_index(drop=True)
    logger.info(
        "HAZ targets computed for %d states: moderate %.1f%% [%.1f–%.1f%%]",
        len(targets),
        targets["haz_moderate_prev"].mean(),
        targets["haz_moderate_prev"].min(),
        targets["haz_moderate_prev"].max(),
    )
    return targets


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dhs_haz_ingestion(cfg: dict) -> pd.DataFrame:
    """
    Called from dimension_targets.py or standalone.
    Reads paths from cfg, saves nga_dhs_haz_targets.csv, returns DataFrame.
    """
    dhs_raw_dir = cfg["paths"].get("dhs_raw_dir",
                                    os.path.join(cfg["paths"]["raw_data_dir"],
                                                  "dhs", "raw"))
    kr_dir  = os.path.join(dhs_raw_dir, "NGKR7BFL")
    gps_shp = os.path.join(dhs_raw_dir, "NGGE7BFL", "NGGE7BFL.shp")

    for p in [kr_dir, gps_shp]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"DHS file/dir not found: {p}")

    targets = compute_haz_state_targets(kr_dir, gps_shp)

    out_path = os.path.join(cfg["paths"]["interim_dir"], "nga_dhs_haz_targets.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    targets.to_csv(out_path, index=False)
    logger.info("DHS HAZ targets saved: %s (%d states)", out_path, len(targets))
    return targets


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Ingest DHS 2018 HAZ nutrition targets for Nigeria")
    parser.add_argument("--country", default="nga")
    args = parser.parse_args()

    from src.utils.config_loader import load_config
    cfg = load_config(f"config/config_{args.country}.yaml")
    targets = run_dhs_haz_ingestion(cfg)
    print(targets.to_string(index=False))
