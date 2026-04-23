"""
Process NBS Nigeria NLSS 2019 state-level poverty headcount rates.

Downloads are from:
  https://nigerianstat.gov.ng/elibrary/read/1092

The NLSS 2019 (household consumption expenditure) gives us a third independent
validation source alongside MICS 2021 (multidimensional child deprivation)
and DHS 2018 (health/nutrition outcomes).

Outputs
-------
Data/Nigeria/nbs/nga_nbs_state_poverty.csv         – clean state-level NBS data
Data/outputs/nga/eval/nbs_model_comparison.csv     – NBS vs model predictions
Data/outputs/nga/eval/nbs_model_comparison.txt     – human-readable report
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EXCEL = ROOT / "Data/Nigeria/nbs/nlss_poverty_2019.xlsx"
NBS_OUT = ROOT / "Data/Nigeria/nbs/nga_nbs_state_poverty.csv"
OUT_DIR = ROOT / "Data/outputs/nga/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# State name harmonization: NBS name → model name (GADM ADM1)
NAME_MAP = {
    "Cross River": "Cross River",
    "FCT":         "Federal Capital Territory",
    "Akwa Ibom":   "Akwa Ibom",
}


def load_nbs() -> pd.DataFrame:
    """Extract state-level poverty headcount from NBS NLSS 2019 Excel."""
    df = pd.read_excel(EXCEL, sheet_name="Poverty Headcount Rate", header=None)
    rows = df.iloc[2:43, [0, 1, 2, 3]].copy()
    rows.columns = ["state", "poverty_headcount_pct", "poverty_gap_pct", "poverty_severity_pct"]
    rows = rows[rows["state"].notna()]
    rows = rows[~rows["state"].isin(["NIGERIA", "Urban", "Rural"])]
    rows["state"] = rows["state"].str.strip()
    rows["state"] = rows["state"].replace(NAME_MAP)
    for col in ["poverty_headcount_pct", "poverty_gap_pct", "poverty_severity_pct"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.dropna(subset=["poverty_headcount_pct"])   # drops Borno (no data)
    rows = rows.reset_index(drop=True)
    return rows


def load_model_predictions() -> pd.DataFrame:
    """Load model state-level predictions (population-weighted from grid).

    Predictions are already in percentage points (0-100 scale).
    """
    pred_path = ROOT / "Data/outputs/nga/tables/nga_predictions.parquet"
    if not pred_path.exists():
        return pd.DataFrame()
    pred = pd.read_parquet(pred_path)

    # Prediction parquet contains 'subregion' for state name
    state_col = next((c for c in ["state", "subregion", "parish_name"] if c in pred.columns), None)
    if state_col is None:
        return pd.DataFrame()
    if state_col != "state":
        pred = pred.rename(columns={state_col: "state"})

    model_cols = [
        c for c in pred.columns
        if c.endswith("_moderate")
        and not c.endswith(("_lower", "_upper"))
        and not c == "moderate_prevalence"
    ]
    pop_col = next((c for c in ["pop_children", "population"] if c in pred.columns), None)

    if not model_cols:
        return pd.DataFrame()

    records = []
    for state, grp in pred.groupby("state"):
        row = {"state": state}
        for col in model_cols:
            if pop_col and grp[pop_col].sum() > 0:
                row[col] = np.average(grp[col].fillna(0), weights=grp[pop_col].clip(0))
            else:
                row[col] = grp[col].mean()
        records.append(row)

    return pd.DataFrame(records)


def build_report(nbs: pd.DataFrame, mics: pd.DataFrame, model: pd.DataFrame) -> str:
    """Build human-readable cross-validation report."""
    # Merge NBS with MICS targets
    merged = nbs.merge(
        mics[["admin_name", "moderate_prevalence"]].rename(columns={"admin_name": "state", "moderate_prevalence": "mics_moderate_pct"}),
        on="state", how="left"
    )

    # Merge with model predictions (if available)
    if not model.empty:
        merged = merged.merge(model, on="state", how="left")
        model_cols = [c for c in model.columns if c != "state"]
    else:
        model_cols = []

    # Sort by NBS poverty rate
    merged = merged.sort_values("poverty_headcount_pct", ascending=False).reset_index(drop=True)

    lines = [
        "=" * 72,
        "NBS NLSS 2019  ×  MICS 2021  ×  Model Predictions",
        "State-Level Cross-Validation — Nigeria",
        "=" * 72,
        "",
        "Three independent data sources measuring poverty at state level:",
        "  NBS NLSS 2019:  household consumption expenditure (monetary poverty)",
        "  MICS 2021:      multidimensional child deprivation (education/health/WASH)",
        "  Model:          spatial prediction from proxy features (RWI, GHSL, etc.)",
        "",
        "NOTE: NBS excludes Borno state (security situation during data collection).",
        "",
    ]

    # NBS vs MICS correlation
    both = merged.dropna(subset=["poverty_headcount_pct", "mics_moderate_pct"])
    if len(both) >= 3:
        rho, p_rho = stats.spearmanr(both["poverty_headcount_pct"], both["mics_moderate_pct"])
        r_val, p_r = stats.pearsonr(both["poverty_headcount_pct"], both["mics_moderate_pct"])
        lines += [
            "-" * 72,
            "NBS vs MICS AGREEMENT",
            "-" * 72,
            f"  Spearman ρ = {rho:+.3f}  (p = {p_rho:.4f})",
            f"  Pearson  r = {r_val:+.3f}  (p = {p_r:.4f})",
            "",
        ]
        if rho >= 0.75:
            lines.append("  Strong agreement — NBS consumption and MICS multidimensional poverty")
            lines.append("  rank states consistently. MICS training targets are well-supported.")
        elif rho >= 0.5:
            lines.append("  Moderate agreement — states broadly consistent; some discrepancies")
            lines.append("  likely due to different poverty concepts (monetary vs multidimensional).")
        else:
            lines.append("  Weak agreement — NBS and MICS rank states differently.")
            lines.append("  May reflect genuine differences between monetary and multidimensional poverty.")
        lines.append("")

    # Main comparison table
    lines += [
        "-" * 72,
        "STATE COMPARISON (sorted by NBS poverty rate)",
        "-" * 72,
    ]
    header = f"{'State':<22} {'NBS%':>7} {'MICS%':>7}"
    if model_cols:
        for mc in model_cols[:3]:
            header += f" {'Model':>8}"
    lines.append(header)
    lines.append("-" * 72)

    for _, row in merged.iterrows():
        nbs_val  = f"{row['poverty_headcount_pct']:.1f}" if pd.notna(row["poverty_headcount_pct"]) else "  N/A"
        mics_val = f"{row['mics_moderate_pct']:.1f}" if pd.notna(row.get("mics_moderate_pct", np.nan)) else "  N/A"
        line = f"{row['state']:<22} {nbs_val:>7} {mics_val:>7}"
        if model_cols:
            for mc in model_cols[:3]:
                v = row.get(mc, np.nan)
                mv = f"{v:.1f}" if pd.notna(v) else "  N/A"
                line += f" {mv:>8}"
        lines.append(line)

    # NBS vs Model correlation
    if model_cols and not model.empty:
        for mc in model_cols[:2]:
            both_m = merged.dropna(subset=["poverty_headcount_pct", mc])
            if len(both_m) >= 3:
                rho_m, p_m = stats.spearmanr(both_m["poverty_headcount_pct"], both_m[mc])
                lines += [
                    "",
                    f"NBS vs {mc}: Spearman ρ = {rho_m:+.3f}  (p = {p_m:.4f})",
                ]

    # Top/bottom 5
    lines += [
        "",
        "-" * 72,
        "TOP 5 MOST DEPRIVED STATES (NBS NLSS 2019)",
        "-" * 72,
    ]
    for _, r in merged.head(5).iterrows():
        lines.append(f"  {r['state']:<22}  {r['poverty_headcount_pct']:.1f}%")

    lines += [
        "",
        "-" * 72,
        "TOP 5 LEAST DEPRIVED STATES (NBS NLSS 2019)",
        "-" * 72,
    ]
    for _, r in merged.tail(5).iterrows():
        lines.append(f"  {r['state']:<22}  {r['poverty_headcount_pct']:.1f}%")

    lines += ["", "=" * 72]
    return "\n".join(lines)


def main():
    print("Loading NBS NLSS 2019 data …")
    nbs = load_nbs()
    nbs.to_csv(NBS_OUT, index=False)
    print(f"  {len(nbs)} states saved → {NBS_OUT}")

    print("Loading MICS targets …")
    mics_path = ROOT / "Data/interim/nga/nga_targets_state.csv"
    mics = pd.read_csv(mics_path).rename(columns={"group_id": "admin_name"}) if mics_path.exists() else pd.DataFrame()

    print("Loading model predictions …")
    model = load_model_predictions()
    if model.empty:
        print("  No model predictions found — skipping model comparison columns")

    report = build_report(nbs, mics, model)
    print("\n" + report)

    out_txt = OUT_DIR / "nbs_model_comparison.txt"
    with open(out_txt, "w") as f:
        f.write(report)

    # Save merged CSV
    merged = nbs.copy()
    if not mics.empty:
        merged = merged.merge(
            mics[["admin_name", "moderate_prevalence"]].rename(columns={"admin_name": "state", "moderate_prevalence": "mics_moderate_pct"}),
            on="state", how="left"
        )
    merged.to_csv(OUT_DIR / "nbs_model_comparison.csv", index=False)
    print(f"\nSaved: {out_txt}")
    print(f"Saved: {OUT_DIR}/nbs_model_comparison.csv")


if __name__ == "__main__":
    main()
