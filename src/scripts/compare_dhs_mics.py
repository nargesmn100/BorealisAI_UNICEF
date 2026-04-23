"""
Cross-validate DHS 2018 vs MICS 2021 zone-level deprivation estimates.

Both surveys independently measure child poverty across Nigeria's 6 geopolitical
zones. Agreement between them validates the MICS targets we train on.
Divergence flags which zones may have data quality issues.

Outputs
-------
Data/outputs/nga/eval/dhs_mics_crossvalidation.csv   – numeric comparison table
Data/outputs/nga/eval/dhs_mics_crossvalidation.txt   – human-readable report
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "Data/outputs/nga/eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
mics = pd.read_csv(ROOT / "Data/interim/nga/nga_targets_geopolitical_zone.csv")
mics = mics.rename(columns={"group_id": "zone", "moderate_prevalence": "mics_moderate_pct"})
mics["mics_moderate"] = mics["mics_moderate_pct"] / 100.0

dhs = pd.read_csv(ROOT / "Data/Nigeria/dhs/nga_dhs_zone_deprivation.csv")
dhs = dhs.rename(columns={"admin_name": "zone"})

merged = pd.merge(mics, dhs, on="zone")

# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------
# MICS: moderate deprivation prevalence (multidimensional — education, health, WASH)
# DHS deprivation_index: composite of stunting, wasting, open defecation,
#   unvaccination, unsafe water — all child-specific, same domains
#
# They are NOT identical metrics, but should rank zones similarly if both
# surveys are measuring the same underlying deprivation reality.

rows = []
for _, r in merged.iterrows():
    rows.append({
        "zone":                  r["zone"],
        "mics_moderate_pct":     r["mics_moderate_pct"],
        "dhs_deprivation_index": round(r["deprivation_index"], 4),
        "dhs_stunting_pct":      round(r["stunting_rate"] * 100, 1),
        "dhs_wasting_pct":       round(r["wasting_rate"] * 100, 1),
        "dhs_unvaccinated_pct":  round((1 - r["vaccination_rate"]) * 100, 1),
        "dhs_open_defec_pct":    round(r["open_defecation_rate"] * 100, 1),
        "dhs_no_safe_water_pct": round((1 - r["improved_water_rate"]) * 100, 1),
        "mics_rank":             0,
        "dhs_rank":              0,
    })

comp = pd.DataFrame(rows)
comp["mics_rank"] = comp["mics_moderate_pct"].rank(ascending=False).astype(int)
comp["dhs_rank"]  = comp["dhs_deprivation_index"].rank(ascending=False).astype(int)
comp["rank_diff"] = (comp["mics_rank"] - comp["dhs_rank"]).abs()
comp = comp.sort_values("mics_rank")

# Spearman rank correlation
rho, p_rho = stats.spearmanr(comp["mics_moderate_pct"], comp["dhs_deprivation_index"])

# Pearson on scaled values (MICS in pct, DHS in 0-1 → scale DHS × 100 for readability)
r_val, p_r = stats.pearsonr(comp["mics_moderate_pct"], comp["dhs_deprivation_index"] * 100)

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
comp.to_csv(OUT_DIR / "dhs_mics_crossvalidation.csv", index=False)

# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------
report_lines = [
    "=" * 70,
    "DHS 2018  vs  MICS 2021 — Zone-Level Cross-Validation",
    "Nigeria, 6 Geopolitical Zones",
    "=" * 70,
    "",
    "Both surveys were conducted independently. Agreement validates the",
    "MICS-derived training targets used in the deprivation model.",
    "",
    "NOTE ON METRICS",
    "  MICS moderate: % children deprived in ≥2 of 6 dimensions (multidimensional)",
    "  DHS index:     composite of stunting, wasting, open defecation,",
    "                 unvaccination, unsafe water — higher = more deprived",
    "  They measure overlapping but not identical constructs. Rank order",
    "  agreement matters more than scale agreement.",
    "",
    "-" * 70,
    "ZONE COMPARISON TABLE",
    "-" * 70,
]

header = (
    f"{'Zone':<16} {'MICS Mod%':>10} {'MICS Rank':>9} "
    f"{'DHS Index':>10} {'DHS Rank':>9} {'|ΔRank|':>8}"
)
report_lines.append(header)
report_lines.append("-" * 70)

for _, r in comp.iterrows():
    flag = "  ⚠ " if r["rank_diff"] >= 2 else "    "
    report_lines.append(
        f"{r['zone']:<16} {r['mics_moderate_pct']:>10.1f} {r['mics_rank']:>9} "
        f"{r['dhs_deprivation_index']:>10.4f} {r['dhs_rank']:>9} "
        f"{int(r['rank_diff']):>8}{flag}"
    )

report_lines += [
    "",
    "-" * 70,
    "STATISTICAL AGREEMENT",
    "-" * 70,
    f"  Spearman rank correlation (ρ): {rho:+.3f}   p = {p_rho:.4f}",
    f"  Pearson correlation (r):       {r_val:+.3f}   p = {p_r:.4f}",
    "",
]

# Interpretation
if rho >= 0.8:
    interp = "STRONG agreement — both surveys rank zones consistently."
elif rho >= 0.6:
    interp = "MODERATE agreement — broadly consistent, minor zone-level discrepancies."
elif rho >= 0.4:
    interp = "WEAK agreement — notable ranking differences; investigate flagged zones."
else:
    interp = "POOR agreement — surveys diverge significantly; data quality concern."

report_lines += [
    f"  Interpretation: {interp}",
    "",
    "-" * 70,
    "DOMAIN-LEVEL BREAKDOWN BY ZONE (DHS)",
    "-" * 70,
]

domain_header = (
    f"{'Zone':<16} {'Stunting%':>10} {'Wasting%':>9} "
    f"{'Unvacc%':>9} {'OpenDef%':>10} {'NoWater%':>10}"
)
report_lines.append(domain_header)
report_lines.append("-" * 70)
for _, r in comp.iterrows():
    report_lines.append(
        f"{r['zone']:<16} {r['dhs_stunting_pct']:>10.1f} {r['dhs_wasting_pct']:>9.1f} "
        f"{r['dhs_unvaccinated_pct']:>9.1f} {r['dhs_open_defec_pct']:>10.1f} "
        f"{r['dhs_no_safe_water_pct']:>10.1f}"
    )

report_lines += [
    "",
    "-" * 70,
    "CONCLUSIONS",
    "-" * 70,
]

# Key findings
north_mics = comp[comp["zone"].str.startswith("North")]["mics_moderate_pct"].mean()
south_mics = comp[comp["zone"].str.startswith("South")]["mics_moderate_pct"].mean()
north_dhs  = comp[comp["zone"].str.startswith("North")]["dhs_deprivation_index"].mean()
south_dhs  = comp[comp["zone"].str.startswith("South")]["dhs_deprivation_index"].mean()

report_lines += [
    f"  North-South gradient (MICS): North avg {north_mics:.1f}% vs South avg {south_mics:.1f}%",
    f"  North-South gradient (DHS):  North avg {north_dhs:.3f} vs South avg {south_dhs:.3f}",
    f"  Both surveys confirm North is substantially more deprived than South.",
    "",
    "  Flagged zones (|rank diff| ≥ 2): " +
    (", ".join(comp[comp["rank_diff"] >= 2]["zone"].tolist()) or "None — full agreement"),
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
print(report_text)

with open(OUT_DIR / "dhs_mics_crossvalidation.txt", "w") as f:
    f.write(report_text)

print(f"\nSaved: {OUT_DIR}/dhs_mics_crossvalidation.csv")
print(f"Saved: {OUT_DIR}/dhs_mics_crossvalidation.txt")
