"""
Per-Dimension Child Deprivation Targets (Kyriaki specification)
===============================================================

Computes individual dimension-level deprivation prevalence per state from
Nigeria MICS6 microdata, as specified by Kyriaki (May 2026):

| Dimension       | Unit                      | Source             | Thresholds                                                   |
|-----------------|---------------------------|--------------------|--------------------------------------------------------------|
| Shelter         | children <17 (hl.sav)    | hh.sav + hl.sav    | moderate ≥3 persons/room; severe ≥5                          |
| Sanitation      | children <17 (hl.sav)    | hh.sav WASH        | moderate improved+shared; severe unimproved                  |
| Water           | children <17 (hl.sav)    | hh.sav WASH        | moderate improved+>30 min; severe unimproved                 |
| Nutrition       | children <5  (ch.sav)    | ch.sav anthro      | moderate HAZ<-2; severe HAZ<-3  [PROXY: MDD, see note]       |
| Education 5–14  | children 5–14 (hl.sav)   | hl.sav education   | moderate not attending; severe never attended                |
| Education 15–17 | youth 15–17 (hl.sav)     | hl.sav education   | moderate not in secondary; severe <primary                   |
| Health 12–35m   | children 12–35m (ch.sav) | ch.sav immuniz.    | moderate missing ≥1 vaccine; severe never vaccinated         |
| Health 36–59m   | children 36–59m (ch.sav) | ch.sav illness/CA  | moderate ARI/fever, no professional care; severe no care     |

Variable mappings (Nigeria MICS6 confirmed from SPSS metadata):
  HC3       — number of rooms used for sleeping  (hh.sav)
  WS1       — drinking water source              (hh.sav)
  WS4       — minutes to collect water           (hh.sav)
  WS11      — toilet facility type               (hh.sav)
  WS14      — toilet facility location           (hh.sav)  3=elsewhere/shared
  HL6       — age of household member            (hl.sav)
  ED4       — ever attended any school/ECE       (hl.sav)  1=yes, 2=no
  ED5A      — highest education level attended   (hl.sav)  11=primary, 21=jnr-sec, 31=snr-sec
  ED10A     — level attended current school year (hl.sav)  null if not attending
  CAGE      — age in months                      (ch.sav)
  IM11      — child ever received any vacc.      (ch.sav)  1=yes, 2=no
  IM20      — child ever given Pentavalent       (ch.sav)  1=yes
  IM21      — times received Pentavalent (DPT)  (ch.sav)  1/2/3
  IM26      — child ever given measles vacc.     (ch.sav)  1=yes
  CA1       — child had diarrhoea in last 2 wks  (ch.sav)  1=yes, 2=no
  CA5       — sought care for diarrhoea          (ch.sav)  1=yes, 2=no
  CA14      — child had fever in last 2 wks      (ch.sav)  1=yes, 2=no
  CA16      — child had cough in last 2 wks      (ch.sav)  1=yes, 2=no
  CA17      — difficulty breathing during cough  (ch.sav)  1=yes, 2=no  (= ARI proxy)
  CA20      — sought any care for ARI/fever      (ch.sav)  1=yes, 2=no
  CA21A/B/C — place sought care: hospital/clinic (ch.sav)  1=yes
  CA21I/J   — place sought care: private hosp/MD (ch.sav)  1=yes

NOTE on Nutrition (HAZ):
  ─────────────────────────────────────────────────────────────────────────────
  Nigeria MICS6 ch.sav does NOT contain HAZ z-scores in its public release.

  IMPLEMENTED FALLBACK HIERARCHY (in order of scientific quality):
    1. DHS 2018 HAZ (NGKR7BFL.DAT, hw70 WHO standard)  ← ACTIVE when available
       State-level stunting targets derived from 11,364 children with valid HAZ.
       Nigeria DHS 2018 mean HAZ = −1.50; stunting 36.2%; severe 16.5%.
       Computed by src/scripts/ingest_dhs_haz.py → nga_dhs_haz_targets.csv.
       Survey timing caveat: DHS 2018 vs MICS 2021 (3-year gap); national
       stunting trend was slowly declining, so rates are conservative.
    2. MDD proxy (< 5 of 8 UNICEF food groups from ch.sav BD8* columns)
       Used only when DHS HAZ file is absent or loading fails.
       This understates stunting and confounds food diversity with growth.

  Future upgrade: if MICS 2021 anthropometry is released separately by
  UNICEF Nigeria / NBS, replace with that file.  Code ready:
      haz_col < -200  (MICS stores z-scores as integer × 100).
  ─────────────────────────────────────────────────────────────────────────────
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Toilet / water classification sets (WHO JMP for Nigeria MICS6 coding)
# ---------------------------------------------------------------------------

# Improved toilet: flush to sewer/septic/pit, VIP pit, pit with slab, composting
IMPROVED_TOILET = {11, 12, 13, 21, 22, 31}
# Unimproved: flush to open drain/unknown, pit without slab, bucket, hanging, open defecation
UNIMPROVED_TOILET = {14, 18, 23, 41, 51, 95}

# Improved water source
IMPROVED_WATER = {11, 12, 13, 14, 21, 31, 41, 51, 91, 92}
# WS4 sentinel values that don't represent real travel times
_WS4_INVALID = {0, 998, 999}

# Secondary education level codes (ED10A / ED5A)
SECONDARY_LEVEL_CODES = {21, 22, 31, 32, 41}   # jnr sec, VEI, snr sec, tech sec, higher


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index)


# ---------------------------------------------------------------------------
# Dimension flag computation — household level (join back to members)
# ---------------------------------------------------------------------------

def _household_size(hl_df: pd.DataFrame) -> pd.DataFrame:
    """Count household members per (HH1, HH2) from hl.sav."""
    sizes = (
        hl_df.groupby(["HH1", "HH2"])
        .size()
        .reset_index(name="hhsize")
    )
    return sizes


def compute_shelter_flags(hh_df: pd.DataFrame, hl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shelter deprivation per household.
    moderate: ≥3 persons per sleeping room
    severe:   ≥5 persons per sleeping room
    Returns one row per household with shelter_moderate, shelter_severe.
    """
    hhsize = _household_size(hl_df)

    hh = hh_df[["HH1", "HH2", "HC3"]].copy()
    hh["HC3"] = pd.to_numeric(hh["HC3"], errors="coerce")
    # HC3 values 45, 8, etc. are likely data entry errors — cap at 20 rooms
    hh["HC3"] = hh["HC3"].clip(lower=1, upper=20)

    hh = hh.merge(hhsize, on=["HH1", "HH2"], how="left")
    hh["ppr"] = hh["hhsize"] / hh["HC3"]

    hh["shelter_moderate"] = (hh["ppr"] >= 3).fillna(False).astype(int)
    hh["shelter_severe"] = (hh["ppr"] >= 5).fillna(False).astype(int)

    logger.info(
        "Shelter flags: moderate=%.1f%%, severe=%.1f%% of households",
        hh["shelter_moderate"].mean() * 100,
        hh["shelter_severe"].mean() * 100,
    )
    return hh[["HH1", "HH2", "shelter_moderate", "shelter_severe"]]


def compute_sanitation_flags(hh_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitation deprivation per household.
    moderate: improved toilet but shared (WS14 == 3 = elsewhere/community facility)
    severe:   unimproved / no facility
    """
    hh = hh_df[["HH1", "HH2", "WS11", "WS14"]].copy()
    ws11 = pd.to_numeric(hh["WS11"], errors="coerce")
    ws14 = pd.to_numeric(hh["WS14"], errors="coerce")

    improved = ws11.isin(IMPROVED_TOILET)
    shared = ws14.eq(3)  # facility elsewhere = shared/community
    unimproved = ws11.isin(UNIMPROVED_TOILET)

    san_mod = (improved & shared).fillna(False).astype(int)
    san_sev = unimproved.fillna(False).astype(int)

    logger.info(
        "Sanitation flags: moderate=%.1f%%, severe=%.1f%% of households",
        san_mod.mean() * 100, san_sev.mean() * 100,
    )
    return pd.DataFrame({
        "HH1": hh["HH1"],
        "HH2": hh["HH2"],
        "sanitation_moderate": san_mod,
        "sanitation_severe": san_sev,
    })


def compute_water_flags(hh_df: pd.DataFrame) -> pd.DataFrame:
    """
    Water deprivation per household.
    moderate: improved source but >30 min roundtrip
    severe:   unimproved / surface / no facility
    """
    hh = hh_df[["HH1", "HH2", "WS1", "WS4"]].copy()
    ws1 = pd.to_numeric(hh["WS1"], errors="coerce")
    ws4 = pd.to_numeric(hh["WS4"], errors="coerce")

    improved = ws1.isin(IMPROVED_WATER)
    valid_time = ws4.notna() & ~ws4.isin(_WS4_INVALID)
    far = valid_time & (ws4 > 30)
    unimproved = ~improved & ws1.notna()

    water_mod = (improved & far).fillna(False).astype(int)
    water_sev = unimproved.fillna(False).astype(int)

    logger.info(
        "Water flags: moderate=%.1f%%, severe=%.1f%% of households",
        water_mod.mean() * 100, water_sev.mean() * 100,
    )
    return pd.DataFrame({
        "HH1": hh["HH1"],
        "HH2": hh["HH2"],
        "water_moderate": water_mod,
        "water_severe": water_sev,
    })


# ---------------------------------------------------------------------------
# Dimension flag computation — individual level (hl.sav members)
# ---------------------------------------------------------------------------

def compute_edu_5_14_flags(hl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Education deprivation for children aged 5–14.
    moderate: not currently attending school (ED10A null/DK for current school year)
    severe:   never attended any school/ECE (ED4 == 2)
    Returns per-member rows; rows outside 5–14 have both flags = 0.
    """
    hl = hl_df.copy()
    age = pd.to_numeric(_safe_col(hl, "HL6"), errors="coerce")
    mask = (age >= 5) & (age <= 14)

    ed4 = _safe_col(hl, "ED4")
    ed10a = pd.to_numeric(_safe_col(hl, "ED10A"), errors="coerce")

    # Not attending: ED10A is null or DK (98)
    not_attending = ed10a.isna() | ed10a.eq(98)
    never_attended = ed4.eq(2)

    mod = (mask & not_attending).fillna(False).astype(int)
    sev = (mask & never_attended).fillna(False).astype(int)

    logger.info(
        "Education 5–14: moderate=%.1f%%, severe=%.1f%% of 5–14 year olds",
        mod[mask].mean() * 100 if mask.any() else 0,
        sev[mask].mean() * 100 if mask.any() else 0,
    )
    return pd.DataFrame({
        "HH1": hl["HH1"],
        "HH2": hl["HH2"],
        "HH6": _safe_col(hl, "HH6"),
        "HH7": _safe_col(hl, "HH7"),
        "hhweight": _safe_col(hl, "hhweight"),
        "age": age,
        "in_group": mask.astype(int),
        "edu_5_14_moderate": mod,
        "edu_5_14_severe": sev,
    })


def compute_edu_15_17_flags(hl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Education deprivation for youth aged 15–17.
    moderate: not currently attending secondary (or higher) AND did not complete secondary
    severe:   did not complete primary
    """
    hl = hl_df.copy()
    age = pd.to_numeric(_safe_col(hl, "HL6"), errors="coerce")
    mask = (age >= 15) & (age <= 17)

    ed5a = pd.to_numeric(_safe_col(hl, "ED5A"), errors="coerce")   # highest level attended
    ed10a = pd.to_numeric(_safe_col(hl, "ED10A"), errors="coerce") # current school year level
    ed4 = _safe_col(hl, "ED4")

    in_secondary = ed10a.isin(SECONDARY_LEVEL_CODES)
    completed_secondary = ed5a >= 21   # junior secondary (21) or higher
    completed_primary = ed5a >= 11     # primary (11) or higher

    # Moderate: not in secondary this year AND didn't reach secondary level
    mod = (mask & ~in_secondary & ~completed_secondary).fillna(False).astype(int)
    # Severe: never completed primary
    sev = (mask & ~completed_primary).fillna(False).astype(int)

    logger.info(
        "Education 15–17: moderate=%.1f%%, severe=%.1f%% of 15–17 year olds",
        mod[mask].mean() * 100 if mask.any() else 0,
        sev[mask].mean() * 100 if mask.any() else 0,
    )
    return pd.DataFrame({
        "HH1": hl["HH1"],
        "HH2": hl["HH2"],
        "HH6": _safe_col(hl, "HH6"),
        "HH7": _safe_col(hl, "HH7"),
        "hhweight": _safe_col(hl, "hhweight"),
        "age": age,
        "in_group": mask.astype(int),
        "edu_15_17_moderate": mod,
        "edu_15_17_severe": sev,
    })


# ---------------------------------------------------------------------------
# Dimension flag computation — child level (ch.sav)
# ---------------------------------------------------------------------------

def compute_health_12_35_flags(ch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Health deprivation for children aged 12–35 months.
    Vaccines required: Pentavalent 1/2/3 (DPT) and Measles.
    moderate: missing ≥1 of the required vaccines
    severe:   never received any vaccination (IM11 == 2)
    """
    ch = ch_df.copy()
    cage = pd.to_numeric(_safe_col(ch, "CAGE"), errors="coerce")
    mask = (cage >= 12) & (cage <= 35)

    im11 = _safe_col(ch, "IM11")   # ever received any vaccination (1=yes, 2=no)
    im20 = _safe_col(ch, "IM20")   # ever given Pentavalent (1=yes)
    im21 = pd.to_numeric(_safe_col(ch, "IM21"), errors="coerce")  # doses of Penta (1/2/3)
    im26 = _safe_col(ch, "IM26")   # ever given measles (1=yes)

    # DPT via Pentavalent counts
    has_dpt1 = im20.eq(1) & im21.ge(1)
    has_dpt2 = im20.eq(1) & im21.ge(2)
    has_dpt3 = im20.eq(1) & im21.ge(3)
    has_measles = im26.eq(1)

    fully_vaccinated = (has_dpt1 & has_dpt2 & has_dpt3 & has_measles).fillna(False)
    never_vaccinated = im11.eq(2).fillna(False)

    mod = (mask & ~fully_vaccinated).fillna(False).astype(int)
    sev = (mask & never_vaccinated).fillna(False).astype(int)

    logger.info(
        "Health 12–35m: moderate=%.1f%%, severe=%.1f%% of 12–35 month olds",
        mod[mask].mean() * 100 if mask.any() else 0,
        sev[mask].mean() * 100 if mask.any() else 0,
    )
    return pd.DataFrame({
        "HH1": ch["HH1"],
        "HH2": ch["HH2"],
        "HH6": _safe_col(ch, "HH6"),
        "HH7": _safe_col(ch, "HH7"),
        "chweight": _safe_col(ch, "chweight"),
        "cage": cage,
        "in_group": mask.astype(int),
        "health_moderate": mod,
        "health_severe": sev,
    })


def compute_health_36_59_flags(ch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Health deprivation for children aged 36–59 months.

    Illness proxy (ARI + fever), care-seeking from a professional provider.
    Kyriaki specification (May 2026):
      Sick  = fever last 2 weeks (CA14==1) OR
               acute respiratory illness: cough (CA16==1) AND difficulty
               breathing (CA17==1) in last 2 weeks.
      moderate: sick but did NOT seek professional care.
                Professional providers = hospital, health centre/clinic,
                private hospital/clinic, private physician
                (CA21A, CA21B, CA21C, CA21I, CA21J == 1).
      severe:   sick but sought NO care at all (CA20 != 1).

    Children who were not sick in the last 2 weeks are flagged as 0 for both.
    """
    ch = ch_df.copy()
    cage = pd.to_numeric(_safe_col(ch, "CAGE"), errors="coerce")
    mask = (cage >= 36) & (cage < 60)

    ca14 = _safe_col(ch, "CA14").eq(1)   # fever
    ca16 = _safe_col(ch, "CA16").eq(1)   # cough
    ca17 = _safe_col(ch, "CA17").eq(1)   # difficulty breathing (ARI)
    ca20 = _safe_col(ch, "CA20")         # sought any care (1=yes, 2=no)

    # Professional facility codes (CA21A–CA21C public, CA21I–CA21J private)
    pro_cols = ["CA21A", "CA21B", "CA21C", "CA21I", "CA21J"]
    saw_professional = pd.Series(False, index=ch.index)
    for col in pro_cols:
        saw_professional = saw_professional | _safe_col(ch, col).eq(1)

    sick = ca14 | (ca16 & ca17)
    any_care = ca20.eq(1)

    mod = (mask & sick & ~(any_care & saw_professional)).fillna(False).astype(int)
    sev = (mask & sick & ~any_care).fillna(False).astype(int)

    logger.info(
        "Health 36–59m: sick=%.1f%%, moderate(no-professional-care)=%.1f%%, "
        "severe(no-care)=%.1f%% of 36–59 month olds",
        sick[mask].mean() * 100 if mask.any() else 0,
        mod[mask].mean() * 100 if mask.any() else 0,
        sev[mask].mean() * 100 if mask.any() else 0,
    )
    return pd.DataFrame({
        "HH1": ch["HH1"],
        "HH2": ch["HH2"],
        "HH6": _safe_col(ch, "HH6"),
        "HH7": _safe_col(ch, "HH7"),
        "chweight": _safe_col(ch, "chweight"),
        "cage": cage,
        "in_group": mask.astype(int),
        "health_36_59_moderate": mod,
        "health_36_59_severe": sev,
    })


def compute_nutrition_flags(ch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Nutrition deprivation for children < 5 years — MDD proxy fallback.

    Used ONLY when DHS HAZ state targets are unavailable (see
    run_nigeria_dimension_targets for the primary HAZ-based path).

    Minimum Dietary Diversity proxy:
      - Moderate: < 5 of 8 UNICEF food groups (BD8 columns)
      - Severe: set to 0 (cannot distinguish without HAZ)
    """
    from src.targets.compute_mics_deprivation import _compute_nutrition_nigeria

    ch = ch_df.copy()
    cage = pd.to_numeric(_safe_col(ch, "CAGE"), errors="coerce")
    mask = cage < 60  # under 5

    dep = _compute_nutrition_nigeria(ch)

    mod = (mask & dep.eq(1)).fillna(False).astype(int)
    sev = mod.copy()
    sev[:] = 0  # cannot distinguish moderate/severe without HAZ

    logger.info(
        "Nutrition (<5y, MDD proxy FALLBACK): moderate=%.1f%% of under-5s",
        mod[mask].mean() * 100 if mask.any() else 0,
    )
    return pd.DataFrame({
        "HH1": ch["HH1"],
        "HH2": ch["HH2"],
        "HH6": _safe_col(ch, "HH6"),
        "HH7": _safe_col(ch, "HH7"),
        "chweight": _safe_col(ch, "chweight"),
        "cage": cage,
        "in_group": mask.astype(int),
        "nutrition_moderate": mod,
        "nutrition_severe": sev,
    })


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _agg_to_state(
    df: pd.DataFrame,
    dim_moderate_col: str,
    dim_severe_col: str,
    weight_col: str,
    state_col: str,
    state_labels: dict | None,
    in_group_col: str | None = None,
) -> list[dict]:
    """
    Weighted aggregation of dimension flags to state level.
    If in_group_col is provided, only rows where in_group == 1 are used as denominator.
    """
    rows = []
    for state_code, grp in df.groupby(state_col):
        if in_group_col:
            grp = grp[grp[in_group_col] == 1]
        if grp.empty:
            continue

        w = pd.to_numeric(grp[weight_col], errors="coerce").fillna(0).values
        total_w = w.sum()
        if total_w == 0:
            continue

        mod_prev = np.average(grp[dim_moderate_col].values, weights=w) * 100
        sev_prev = np.average(grp[dim_severe_col].values, weights=w) * 100

        if state_labels and state_code in state_labels:
            subregion = state_labels[state_code].strip().title()
        else:
            try:
                subregion = str(int(state_code))
            except (ValueError, TypeError):
                subregion = str(state_code)

        rows.append({
            "subregion": subregion,
            f"{dim_moderate_col.replace('_moderate', '')}_moderate_prev": round(mod_prev, 2),
            f"{dim_moderate_col.replace('_moderate', '')}_severe_prev": round(sev_prev, 2),
            f"{dim_moderate_col.replace('_moderate', '')}_sample_n": int(grp[in_group_col].sum()) if in_group_col else len(grp),
        })
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_kyriaki_dimension_targets(
    ch_df: pd.DataFrame,
    hh_df: pd.DataFrame,
    hl_df: pd.DataFrame,
    state_labels: dict | None = None,
) -> pd.DataFrame:
    """
    Compute per-state prevalence for all 8 Kyriaki dimensions.

    Returns a DataFrame with columns:
        subregion,
        shelter_moderate_prev,      shelter_severe_prev,
        sanitation_moderate_prev,   sanitation_severe_prev,
        water_moderate_prev,        water_severe_prev,
        nutrition_moderate_prev,    nutrition_severe_prev,
        edu_5_14_moderate_prev,     edu_5_14_severe_prev,
        edu_15_17_moderate_prev,    edu_15_17_severe_prev,
        health_moderate_prev,       health_severe_prev,
        health_36_59_moderate_prev, health_36_59_severe_prev
    """
    logger.info("Computing Kyriaki dimension targets (8 dimensions)...")

    # ── 1. Household-level dimensions: join flags back to all hl.sav members ──
    shelter_hh = compute_shelter_flags(hh_df, hl_df)
    sanitation_hh = compute_sanitation_flags(hh_df)
    water_hh = compute_water_flags(hh_df)

    # Merge household flags onto all hl.sav members (each member inherits HH conditions)
    hl_merged = hl_df[["HH1", "HH2", "HL6", "HH6", "HH7", "hhweight"]].copy()
    hl_merged["HL6"] = pd.to_numeric(hl_merged["HL6"], errors="coerce")
    hl_merged = hl_merged[hl_merged["HL6"] < 17]  # children < 17 per spec

    for flags_df in [shelter_hh, sanitation_hh, water_hh]:
        hl_merged = hl_merged.merge(flags_df, on=["HH1", "HH2"], how="left")

    hl_merged["in_group"] = 1  # all members under 17

    # ── 2. Education dimensions (hl.sav, age-filtered internally) ──
    edu_5_14 = compute_edu_5_14_flags(hl_df)
    edu_15_17 = compute_edu_15_17_flags(hl_df)

    # ── 3. Child-level dimensions (ch.sav) ──
    health = compute_health_12_35_flags(ch_df)
    health_36_59 = compute_health_36_59_flags(ch_df)
    nutrition = compute_nutrition_flags(ch_df)

    # ── 4. Aggregate each dimension to state level ──
    state_col = "HH7"

    all_states = sorted(
        set(hl_merged[state_col].dropna().unique()) |
        set(edu_5_14[state_col].dropna().unique()) |
        set(health[state_col].dropna().unique()) |
        set(health_36_59[state_col].dropna().unique())
    )

    # Collect per-state rows
    results: dict[str, dict] = {}

    def _init_state(code):
        if state_labels and code in state_labels:
            name = state_labels[code].strip().title()
        else:
            try:
                name = str(int(code))
            except (ValueError, TypeError):
                name = str(code)
        return {"subregion": name}

    for code in all_states:
        results[code] = _init_state(code)

    # Helper to fill results for each dimension
    def _fill_dim(flags_df, mod_col, sev_col, wt_col, in_grp_col=None):
        for code, grp in flags_df.groupby(state_col):
            if in_grp_col:
                grp = grp[grp[in_grp_col] == 1]
            if grp.empty or code not in results:
                continue
            w = pd.to_numeric(grp[wt_col], errors="coerce").fillna(0).values
            if w.sum() == 0:
                continue
            dim_name = mod_col.replace("_moderate", "")
            results[code][f"{dim_name}_moderate_prev"] = round(
                np.average(grp[mod_col].values, weights=w) * 100, 2
            )
            results[code][f"{dim_name}_severe_prev"] = round(
                np.average(grp[sev_col].values, weights=w) * 100, 2
            )
            results[code][f"{dim_name}_n"] = int(len(grp))

    _fill_dim(hl_merged, "shelter_moderate", "shelter_severe", "hhweight", "in_group")
    _fill_dim(hl_merged, "sanitation_moderate", "sanitation_severe", "hhweight", "in_group")
    _fill_dim(hl_merged, "water_moderate", "water_severe", "hhweight", "in_group")
    _fill_dim(edu_5_14, "edu_5_14_moderate", "edu_5_14_severe", "hhweight", "in_group")
    _fill_dim(edu_15_17, "edu_15_17_moderate", "edu_15_17_severe", "hhweight", "in_group")
    _fill_dim(health, "health_moderate", "health_severe", "chweight", "in_group")
    _fill_dim(health_36_59, "health_36_59_moderate", "health_36_59_severe", "chweight", "in_group")
    _fill_dim(nutrition, "nutrition_moderate", "nutrition_severe", "chweight", "in_group")

    targets = pd.DataFrame(list(results.values()))
    targets = targets.sort_values("subregion").reset_index(drop=True)

    # Log summary
    dim_cols = [c for c in targets.columns if c.endswith("_moderate_prev")]
    logger.info("Dimension targets computed for %d states:", len(targets))
    for col in dim_cols:
        dim = col.replace("_moderate_prev", "")
        vals = targets[col].dropna()
        if not vals.empty:
            logger.info(
                "  %-20s moderate: [%.1f%%, %.1f%%] mean=%.1f%%",
                dim, vals.min(), vals.max(), vals.mean()
            )

    return targets


def _load_dhs_haz_targets(cfg: dict) -> "pd.DataFrame | None":
    """
    Try to load DHS HAZ state targets.  Returns None if unavailable.

    Looks first for a pre-computed CSV, then runs ingest_dhs_haz if raw
    DHS KR files are present.
    """
    interim_dir = cfg["paths"]["interim_dir"]
    cached_csv = os.path.join(interim_dir, "nga_dhs_haz_targets.csv")

    if os.path.isfile(cached_csv):
        df = pd.read_csv(cached_csv)
        logger.info("DHS HAZ targets loaded from cache: %s (%d states)", cached_csv, len(df))
        return df

    # Try to compute from raw files
    dhs_raw_dir = cfg["paths"].get(
        "dhs_raw_dir",
        os.path.join(cfg["paths"]["data_dir"], "Nigeria", "dhs", "raw"),
    )
    kr_dir  = os.path.join(dhs_raw_dir, "NGKR7BFL")
    gps_shp = os.path.join(dhs_raw_dir, "NGGE7BFL", "NGGE7BFL.shp")

    if not (os.path.isdir(kr_dir) and os.path.isfile(gps_shp)):
        logger.warning(
            "DHS KR files not found at %s — falling back to MDD proxy for nutrition",
            kr_dir,
        )
        return None

    try:
        from src.scripts.ingest_dhs_haz import compute_haz_state_targets
        targets = compute_haz_state_targets(kr_dir, gps_shp)
        os.makedirs(interim_dir, exist_ok=True)
        targets.to_csv(cached_csv, index=False)
        logger.info("DHS HAZ targets computed and cached: %s", cached_csv)
        return targets
    except Exception as exc:
        logger.warning("DHS HAZ ingestion failed (%s) — falling back to MDD proxy", exc)
        return None


def run_nigeria_dimension_targets(cfg: dict) -> pd.DataFrame:
    """
    Entry point: load MICS6 SPSS files and compute all 8 dimension targets.

    Nutrition dimension:
      - Primary:  DHS 2018 HAZ-based stunting (NGKR7BFL.DAT, hw70 WHO)
      - Fallback: MICS MDD proxy (when DHS KR not available)

    Saves `nga_dimension_targets.csv` to `cfg.paths.interim_dir`.
    Returns the dimension targets DataFrame.
    """
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "pyreadstat is required for MICS SPSS data. "
            "Install with: pip install pyreadstat"
        )

    mics_dir = cfg["paths"]["mics_data_dir"]
    logger.info("Loading MICS6 SPSS files for dimension targets from: %s", mics_dir)

    ch_path = os.path.join(mics_dir, "ch.sav")
    hh_path = os.path.join(mics_dir, "hh.sav")
    hl_path = os.path.join(mics_dir, "hl.sav")

    for p in [ch_path, hh_path, hl_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required MICS file missing: {p}")

    logger.info("Loading ch.sav...")
    ch_df, ch_meta = pyreadstat.read_sav(ch_path)
    logger.info("  ch.sav: %d rows", len(ch_df))

    logger.info("Loading hh.sav...")
    hh_df, hh_meta = pyreadstat.read_sav(hh_path)
    logger.info("  hh.sav: %d rows", len(hh_df))

    logger.info("Loading hl.sav...")
    hl_df, hl_meta = pyreadstat.read_sav(hl_path)
    logger.info("  hl.sav: %d rows", len(hl_df))

    # Resolve state labels from SPSS metadata
    state_labels = {}
    for meta in [ch_meta, hh_meta, hl_meta]:
        if "HH7" in meta.variable_value_labels:
            state_labels = {
                int(k) if isinstance(k, float) else k: v
                for k, v in meta.variable_value_labels["HH7"].items()
            }
            break
    if state_labels:
        logger.info("State labels resolved: %d states", len(state_labels))

    targets = compute_kyriaki_dimension_targets(ch_df, hh_df, hl_df, state_labels)

    # ── Replace MDD nutrition with DHS HAZ if available ──────────────────────
    haz = _load_dhs_haz_targets(cfg)
    if haz is not None:
        logger.info(
            "Replacing MDD nutrition target with DHS 2018 HAZ stunting "
            "(%d states available).", len(haz)
        )
        haz_merge = haz.rename(columns={
            "haz_moderate_prev": "nutrition_moderate_prev",
            "haz_severe_prev":   "nutrition_severe_prev",
            "haz_n":             "nutrition_n",
        })
        targets = targets.drop(
            columns=[c for c in targets.columns if c.startswith("nutrition_")],
            errors="ignore",
        )
        targets = targets.merge(haz_merge, on="subregion", how="left")
        n_merged = targets["nutrition_moderate_prev"].notna().sum()
        logger.info(
            "  HAZ nutrition merged: %d / %d states (mean moderate stunting: %.1f%%)",
            n_merged, len(targets), targets["nutrition_moderate_prev"].mean(),
        )
        if n_merged < len(targets):
            missing_states = targets.loc[
                targets["nutrition_moderate_prev"].isna(), "subregion"
            ].tolist()
            logger.warning(
                "  States without HAZ data (NaN nutrition target): %s", missing_states
            )
    else:
        logger.info("Using MDD proxy for nutrition (DHS HAZ not available).")

    out_path = os.path.join(
        cfg["paths"]["interim_dir"], "nga_dimension_targets.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    targets.to_csv(out_path, index=False)
    logger.info(
        "Dimension targets saved to: %s (%d states × %d columns)",
        out_path, len(targets), len(targets.columns)
    )

    return targets
