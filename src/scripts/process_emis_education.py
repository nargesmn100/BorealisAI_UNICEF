"""
Extract state-level school attendance and education quality indicators
from MICS6 2021 Nigeria household listing microdata (hl.sav).

These indicators serve as features for the deprivation model, capturing
school utilization quality beyond just distance to nearest school (dist_school_km).

Key indicators computed:
  school_attendance_rate   – fraction of school-age children (6-17) currently attending
  school_attendance_rural  – same, rural only
  school_attendance_urban  – same, urban only
  public_school_rate       – fraction attending public (vs private) school
  ever_attended_rate       – fraction who ever attended school (6+ years old)

Outputs
-------
Data/Nigeria/features/education/nga_mics_education_by_state.csv
Data/Nigeria/features/education/nga_mics_education_by_state_urbanrural.csv
"""

import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HL_SAV = ROOT / "Data/Nigeria/Nigeria MICS6 SPSS Datasets/hl.sav"
OUT_DIR = ROOT / "Data/Nigeria/features/education"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCHOOL_AGE_MIN = 6
SCHOOL_AGE_MAX = 17

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


def load_hl() -> pd.DataFrame:
    """Load household listing file with education variables."""
    print("Loading MICS hl.sav …")
    df, meta = pyreadstat.read_sav(
        HL_SAV,
        usecols=["HH1", "HH6", "HH7", "HL6", "HL4",
                 "ED4",   # ever attended school
                 "ED9",   # attended school current year
                 "ED11",  # attended public school
                 "hhweight"],
    )
    print(f"  {len(df):,} household members loaded")

    df["state"] = df["HH7"].map(STATE_LABELS)
    df["urban_rural"] = df["HH6"].map({1.0: "urban", 2.0: "rural"})
    df["weight"] = df["hhweight"].fillna(0)

    # Filter to school-age children
    df["age"] = pd.to_numeric(df["HL6"], errors="coerce")
    school_age = df[(df["age"] >= SCHOOL_AGE_MIN) & (df["age"] <= SCHOOL_AGE_MAX)].copy()
    print(f"  School-age members (age {SCHOOL_AGE_MIN}–{SCHOOL_AGE_MAX}): {len(school_age):,}")

    # Recode indicators
    # ED9: 1=yes attended this year, 2=no, 9=DK
    school_age["attends_school"] = school_age["ED9"].map({1.0: 1.0, 2.0: 0.0}).astype(float)
    # ED4: 1=yes ever, 2=no
    school_age["ever_attended"] = school_age["ED4"].map({1.0: 1.0, 2.0: 0.0}).astype(float)
    # ED11: 1=public, 2=private — only for those attending
    school_age["attends_public"] = school_age["ED11"].map({1.0: 1.0, 2.0: 0.0}).astype(float)

    return school_age


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    valid = series.notna() & weights.notna() & (weights > 0)
    if valid.sum() == 0:
        return np.nan
    return np.average(series[valid], weights=weights[valid])


def aggregate(df: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    """Compute weighted education indicators by groupby columns."""
    records = []
    for keys, grp in df.groupby(groupby):
        if isinstance(keys, str):
            keys = (keys,)
        row = dict(zip(groupby, keys))
        row["n_children"] = len(grp)
        row["school_attendance_rate"] = weighted_mean(grp["attends_school"], grp["weight"])
        row["ever_attended_rate"]     = weighted_mean(grp["ever_attended"],  grp["weight"])
        # public_school_rate only among attendees
        attendees = grp[grp["attends_school"] == 1]
        row["public_school_rate"] = weighted_mean(attendees["attends_public"], attendees["weight"])
        records.append(row)
    return pd.DataFrame(records)


def main():
    df = load_hl()

    print("\nComputing state-level education indicators …")
    state_df = aggregate(df, ["state"])
    state_df = state_df.sort_values("school_attendance_rate").reset_index(drop=True)

    out_state = OUT_DIR / "nga_mics_education_by_state.csv"
    state_df.to_csv(out_state, index=False)
    print(f"  Saved: {out_state}  ({len(state_df)} states)")

    print("\nComputing state × urban/rural education indicators …")
    ur_df = aggregate(df, ["state", "urban_rural"])
    out_ur = OUT_DIR / "nga_mics_education_by_state_urbanrural.csv"
    ur_df.to_csv(out_ur, index=False)
    print(f"  Saved: {out_ur}  ({len(ur_df)} rows)")

    print("\n=== School Attendance Rate by State ===")
    print(f"{'State':<28} {'Attendance%':>12} {'EverAttended%':>14} {'Public%':>9}")
    print("-" * 67)
    for _, r in state_df.sort_values("school_attendance_rate", ascending=False).iterrows():
        att = f"{r['school_attendance_rate']*100:.1f}" if pd.notna(r['school_attendance_rate']) else " N/A"
        ever = f"{r['ever_attended_rate']*100:.1f}" if pd.notna(r['ever_attended_rate']) else " N/A"
        pub = f"{r['public_school_rate']*100:.1f}" if pd.notna(r['public_school_rate']) else " N/A"
        print(f"{r['state']:<28} {att:>12} {ever:>14} {pub:>9}")

    print("\n=== Bottom 5 States by Attendance ===")
    for _, r in state_df.head(5).iterrows():
        print(f"  {r['state']:<28}  {r['school_attendance_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
