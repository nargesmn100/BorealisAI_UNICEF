# Current status — May 4, 2026 (final evening update)

> **Research tool only.** Outputs are not official poverty statistics.  
> UNICEF × RBC Borealis AI — Nigeria child deprivation pipeline.

This snapshot summarises the repository as of **4 May 2026 (evening)**. The living master checklist remains **`PROJECT_STATUS.md`** (update that file whenever you ship substantive changes).

---

## Executive summary

The Nigeria pipeline is **end-to-end operational and science-grade**:

- **103,424-cell** grid at ~2.4 km (RWI-aligned) covering all of Nigeria
- **8 deprivation dimensions** (Shelter, Sanitation, Water, Nutrition, Edu 5–14, Edu 15–17, Health 12–35m, **Health 36–59m** — new)
- **Nutrition target upgraded**: DHS 2018 HAZ stunting (41.0%) replaces the MDD proxy (27.2%) via `ingest_dhs_haz.py` parsing `NGKR7BFL.DAT`
- **Predictor leakage removed**: 8 MICS survey-rate features that overlapped targets were dropped; replaced with geospatial non-survey signals
- **DHS stacked auxiliary Ridge** training at `scale=1.0` — Spearman ρ vs DHS GPS clusters: **0.553**
- **D1 external features**: NBS MPI (9), NEMIS (7), IIAG (stored, not in training) — **38 total training features**
- **LGA rollup**: 775 LGAs with population-weighted predictions for all 8 dimensions
- **9 interactive dimension maps** generated (individual + comparison panels)
- **All evaluation tables refreshed** (LOZO, hierarchical CV, dimension summary)

---

## Grid and supervision

| Item | Detail |
|------|--------|
| Base geometry | **103,424** points — RWI Nigeria grid (~**2.4 km** cell spacing) |
| Primary targets | MICS6 composite moderate/severe + **8 dimension-level** prevalences (37 states) |
| Nutrition target | **DHS 2018 HAZ stunting** (`NGKR7BFL.DAT`, 11,364 child records) |
| Ridge training signal | MICS block + DHS nearest-cluster auxiliary block (`dhs_aux_mics_scale: 1.0`, `dhs_aux_dhs_scale: 1.0`) |
| Population | WorldPop 2020 constrained raster (under-5) |

---

## Features (`modeling.features`)

| Block | Count | Notes |
|-------|------:|------|
| Non-survey geospatial | **22** | RWI, travel time, nightlights, rainfall, ACLED, OSM, GHSL, SMOD, DHS nearest-cluster (2) |
| D1 external (state-level) | **16** | NBS MPI (9) + NEMIS (7 key columns) |
| **Total training features** | **38** | Leaky MICS rates removed (8 features dropped from prior 46) |
| IIAG | 7 on modeling table | Excluded — national scalars with no within-Nigeria variation |

> **Leakage removal:** `school_attendance_rate`, `ever_attended_rate`, `public_school_rate`, `anc_rate`, `skilled_delivery_rate`, `facility_delivery_rate`, `vacc_card_rate`, `diarrhea_care_rate` removed. Ridge Pearson r: **0.40** (was ~0.55 inflated).

---

## Per-dimension pipeline (Kyriaki specification)

| Dimension | Status | Target source | Mean prevalence |
|-----------|--------|---------------|----------------|
| Shelter (overcrowding) | ✅ | MICS6 `hh.sav` + `hl.sav` | 56.1% |
| Sanitation | ✅ | MICS6 `hh.sav` | 2.0% |
| Water | ✅ | MICS6 `hh.sav` | 4.3% |
| Nutrition | ✅ **DHS HAZ** | **DHS 2018 `NGKR7BFL.DAT`** | **41.0%** (was 27.2% MDD) |
| Education 5–14 | ✅ | MICS6 `hl.sav` | 33.6% |
| Education 15–17 | ✅ | MICS6 `hl.sav` | 37.4% |
| Health 12–35m (vaccines) | ✅ | MICS6 `ch.sav` | 89.2% |
| **Health 36–59m (ARI+care)** | ✅ **new** | MICS6 `ch.sav` | **22.8%** |

---

## Models and key outputs

| Model | Role |
|-------|------|
| Ridge | Primary interpretable model; DHS aux-stack; breakdown/themes |
| GBM, GAM, WSNN | Optional; WSNN permutation importance → `nga_wsnn_importance.csv` |
| Baselines | Uniform, heuristic, RWI redistribution |

**Representative artifacts** (under `Data/outputs/nga/`):

- `tables/nga_predictions.parquet`, `nga_lga_predictions.csv`, `nga_dimension_predictions.csv`
- `maps/nga_predictions_map_sample.html`, `nga_comparison_map.html`
- `maps/nga_dimension_{shelter,sanitation,water,nutrition,edu_5_14,edu_15_17,health,health_36_59}_map.html` — 8 individual maps
- `eval/evaluation_summary.csv`, `lozo_evaluation.csv`, `nga_dimension_summary.csv`, `hierarchical_validation.csv`

---

## Validation snapshot (May 4, 2026 — leakage-free)

| Metric | Value |
|--------|-------|
| LOZO MAE (Ridge) | ~12.7 pp |
| LOZO MAE (WSNN) | ~6.1 pp (best) |
| LOZO MAE (GBM) | ~7.4 pp |
| Composite Pearson r (Ridge, leakage-free) | **0.40** |
| RWI redistribution baseline Pearson r | **0.79** |
| Hierarchical CV (zone→state, Ridge r) | **0.85** |
| DHS GPS Spearman ρ (stacked aux 1.0) | **0.553** |
| Nutrition train_r (DHS HAZ) | **0.706** |

---

## Stakeholder feedback (May 2026) — status

| Topic | Status |
|-------|--------|
| Individual dimension targets | ✅ 8 dimensions implemented |
| HAZ nutrition | ✅ DHS 2018 NGKR7BFL ingested |
| Second health age band (36–59m) | ✅ Implemented |
| Remove MICS leakage predictors | ✅ Done |
| Grid wording (~2.4 km vs "~1 km²") | ⚠️ `TECHNICAL_OVERVIEW.md` alignment pending |
| WorldPop age bands | Pending |
| MICS GIS clusters 2021 | Pending |
| D2 second country | Pending |

---

## Recommended next steps

1. **DHS aux LOZO validation** — Run `python src/scripts/dhs_aux_sweep.py` (without `--skip-lozo`) to confirm `dhs_aux_dhs_scale=1.0` does not harm held-out states (~2–3 hours).
2. **`TECHNICAL_OVERVIEW.md` wording** — Verify and fix any stray "~1 km²" grid references.
3. **D2 second country** — `config_{iso}.yaml` + MICS + geo data; proves portability.
4. **NEMIS/NBS LGA harmonisation** — Match LGA strings to GADM ADM2 for sub-state validation.
5. **U1 stakeholder default** — Adopt `nga_comparison_map.html` (LGA polygons) as the default shared artifact.
6. **GBM SHAP** (optional) — Enable `export_gbm_shap: true` for stakeholder explainability.

---

## Git / GitHub — large files (`GH001`)

A push to GitHub was **rejected** when commits contained generated maps and CSVs over **100 MB**. Recovery was performed via `git-filter-repo` to rewrite `franklin-2` branch history. **`.gitignore`** excludes full-grid Folium HTML files, large prediction CSVs, and NBS MPI `.dta` files. See `PROJECT_STATUS.md §4` for full recovery instructions.

---

## Where to read more

| Document | Purpose |
|----------|---------|
| `PROJECT_STATUS.md` | Full checklist, priorities 1–6, completed-since list, master concerns |
| `TECHNICAL_OVERVIEW.md` | Data sources, column reference, architecture |
| `HIGH_LEVEL_MODEL_OVERVIEW.md` | Model narrative, Ridge decomposition, breakdown schema |
| `config/config_nga.yaml` | Paths, `modeling.features`, Ridge DHS knobs, dimensions `feature_overrides` |
| `Data/Nigeria/d1_external/README.md` | D1 raw data layout and re-download notes |
| `src/scripts/ingest_dhs_haz.py` | DHS 2018 HAZ ingestion — `.DAT` parsing, cluster→state, stunting targets |
| `src/targets/dimension_targets.py` | All 8 dimension flag definitions; HAZ fallback hierarchy |

---

*End of snapshot — May 4, 2026 (evening, all major tasks complete).*
