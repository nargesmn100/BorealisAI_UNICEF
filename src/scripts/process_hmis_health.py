"""
Extract state-level health utilization indicators from MICS6 2021 Nigeria.

Sources:
  wm.sav  — Women's questionnaire (ANC, skilled delivery, facility birth)
  ch.sav  — Children under-5 file (vaccination card, diarrhoea care-seeking)

These indicators capture health *system utilization* beyond just proximity
(dist_health_km), providing a more direct measure of health access quality.

Key indicators:
  anc_rate              – fraction of women who received any prenatal care
  skilled_delivery_rate – fraction delivered by doctor or nurse/midwife
  facility_delivery_rate– fraction who delivered in a health facility
  vacc_card_rate        – fraction of children with a vaccination card seen
  diarrhea_treatment_rate – fraction who sought care for child diarrhea

Outputs
-------
Data/Nigeria/features/health/nga_mics_health_by_state.csv
Data/Nigeria/features/health/nga_mics_health_by_state_urbanrural.csv
"""

import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MICS_DIR = ROOT / "Data/Nigeria/Nigeria MICS6 SPSS Datasets"
OUT_DIR = ROOT / "Data/Nigeria/features/health"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# MN20 facility delivery codes (home=11,12; facility=21-36)
FACILITY_DELIVERY_CODES = {21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 31.0, 32.0, 33.0, 34.0, 36.0}


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    valid = series.notna() & weights.notna() & (weights > 0)
    if valid.sum() == 0:
        return np.nan
    return float(np.average(series[valid], weights=weights[valid]))


# ---------------------------------------------------------------------------
# Women's file — ANC and delivery indicators
# ---------------------------------------------------------------------------
def load_women() -> pd.DataFrame:
    print("Loading wm.sav (women's questionnaire) …")
    df, meta = pyreadstat.read_sav(
        MICS_DIR / "wm.sav",
        usecols=["HH1", "HH6", "HH7",
                 "MN2",                         # received any ANC
                 "MN5",                         # number of ANC visits
                 "MN19A", "MN19B", "MN19C",    # skilled attendant (Dr / nurse / CHEW)
                 "MN20",                        # place of delivery
                 "wmweight"],
    )
    print(f"  {len(df):,} women loaded")

    df["state"]      = df["HH7"].map(STATE_LABELS)
    df["urban_rural"] = df["HH6"].map({1.0: "urban", 2.0: "rural"})
    df["weight"]     = df["wmweight"].fillna(0)

    # ANC: MN2 == 1 means yes
    df["had_anc"] = df["MN2"].map({1.0: 1.0, 2.0: 0.0})

    # 4+ ANC visits
    df["anc_4plus"] = (pd.to_numeric(df["MN5"], errors="coerce") >= 4).astype(float)
    df["anc_4plus"] = df["anc_4plus"].where(df["had_anc"] == 1, other=0.0)

    # Skilled delivery: MN19A='A' (doctor), MN19B='B' (nurse/midwife), MN19C='C' (CHEW)
    # Variables are string-coded — non-empty letter = mentioned
    skilled_codes = {"MN19A": "A", "MN19B": "B", "MN19C": "C"}
    df["skilled_delivery"] = df.apply(
        lambda row: 1.0 if any(
            str(row.get(col, "")).strip() == code
            for col, code in skilled_codes.items()
        ) else 0.0,
        axis=1,
    )

    # Facility delivery: MN20 code in facility set
    df["facility_delivery"] = (
        pd.to_numeric(df["MN20"], errors="coerce").isin(FACILITY_DELIVERY_CODES)
    ).astype(float)

    return df


# ---------------------------------------------------------------------------
# Children's file — vaccination card and diarrhea care-seeking
# ---------------------------------------------------------------------------
def load_children() -> pd.DataFrame:
    print("Loading ch.sav (children under-5) …")
    df, meta = pyreadstat.read_sav(
        MICS_DIR / "ch.sav",
        usecols=["HH1", "HH6", "HH7",
                 "IM2",   # vaccination card seen (1=yes seen, 2=yes not seen, 3=no)
                 "CA5",   # sought care for diarrhea (1=yes, 2=no)
                 "chweight"],
    )
    print(f"  {len(df):,} children loaded")

    df["state"]       = df["HH7"].map(STATE_LABELS)
    df["urban_rural"] = df["HH6"].map({1.0: "urban", 2.0: "rural"})
    df["weight"]      = df["chweight"].fillna(0)

    # vaccination card seen or reported (codes 1 or 2 = has card)
    df["has_vacc_card"] = df["IM2"].map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0})

    # diarrhea care-seeking
    df["diarrhea_care"] = df["CA5"].map({1.0: 1.0, 2.0: 0.0})

    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame, ind_cols: list[str],
              weight_col: str, groupby: list[str]) -> pd.DataFrame:
    records = []
    for keys, grp in df.groupby(groupby):
        if isinstance(keys, str):
            keys = (keys,)
        row = dict(zip(groupby, keys))
        row["n"] = len(grp)
        for col in ind_cols:
            row[col] = weighted_mean(grp[col], grp[weight_col])
        records.append(row)
    return pd.DataFrame(records)


def main():
    # Women's indicators
    wm = load_women()
    wm_inds = ["had_anc", "anc_4plus", "skilled_delivery", "facility_delivery"]

    wm_state = aggregate(wm, wm_inds, "weight", ["state"])
    wm_ur    = aggregate(wm, wm_inds, "weight", ["state", "urban_rural"])

    # Children's indicators
    ch = load_children()
    ch_inds = ["has_vacc_card", "diarrhea_care"]

    ch_state = aggregate(ch, ch_inds, "weight", ["state"])
    ch_ur    = aggregate(ch, ch_inds, "weight", ["state", "urban_rural"])

    # Merge
    state_df = wm_state.merge(ch_state[["state"] + ch_inds], on="state", how="outer")
    ur_df    = wm_ur.merge(ch_ur[["state", "urban_rural"] + ch_inds],
                           on=["state", "urban_rural"], how="outer")

    # Rename for clarity
    rename = {
        "had_anc":           "anc_rate",
        "anc_4plus":         "anc_4plus_rate",
        "skilled_delivery":  "skilled_delivery_rate",
        "facility_delivery": "facility_delivery_rate",
        "has_vacc_card":     "vacc_card_rate",
        "diarrhea_care":     "diarrhea_care_rate",
    }
    state_df = state_df.rename(columns=rename)
    ur_df    = ur_df.rename(columns=rename)

    out_state = OUT_DIR / "nga_mics_health_by_state.csv"
    out_ur    = OUT_DIR / "nga_mics_health_by_state_urbanrural.csv"
    state_df.to_csv(out_state, index=False)
    ur_df.to_csv(out_ur, index=False)
    print(f"\nSaved: {out_state}  ({len(state_df)} states)")
    print(f"Saved: {out_ur}  ({len(ur_df)} rows)")

    # Summary table
    ind_cols = list(rename.values())
    print("\n=== Health Utilization by State (sorted by skilled delivery) ===")
    print(f"{'State':<28} {'ANC%':>6} {'ANC4+%':>7} {'Skill%':>7} {'Facil%':>7} {'VaccCard%':>10} {'DiaCare%':>9}")
    print("-" * 80)
    for _, r in state_df.sort_values("skilled_delivery_rate", ascending=False).iterrows():
        def fmt(v):
            return f"{v*100:.1f}" if pd.notna(v) else " N/A"
        print(f"{r['state']:<28} {fmt(r['anc_rate']):>6} {fmt(r['anc_4plus_rate']):>7} "
              f"{fmt(r['skilled_delivery_rate']):>7} {fmt(r['facility_delivery_rate']):>7} "
              f"{fmt(r['vacc_card_rate']):>10} {fmt(r['diarrhea_care_rate']):>9}")


if __name__ == "__main__":
    main()
