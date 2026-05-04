# Current status — May 4, 2026

> **Research tool only.** Outputs are not official poverty statistics.  
> UNICEF × RBC Borealis AI — Nigeria child deprivation pipeline.

This snapshot summarises the repository as of **4 May 2026**. The living master checklist remains **`PROJECT_STATUS.md`** (update that file whenever you ship substantive changes).

---

## Executive summary

The Nigeria pipeline is **end-to-end operational**: a **103,424-cell** grid aligned to the **RWI** product (~**2.4 km** cell spacing), **state-level MICS6 2021** supervision for composite moderate/severe deprivation, **Ridge** training with **DHS stacked auxiliary** loss (`dhs_aux_dhs_scale: 1.0`), optional **GBM / GAM / WSNN**, rich **evaluation** (LOZO, hierarchical CV, DHS GPS checks), **Folium** maps (clustering, sampled export, legends, Ridge explain snippets), **LGA** population-weighted rollups (**775 LGAs**), and **D1** external features (**NBS MPI + NEMIS**, state-joined; **IIAG** on the table but not in the training feature list).

**Per-dimension (Kyriaki-style) targets and Ridge models** exist for **seven** dimensions; nutrition uses an **MDD proxy** (no HAZ in Nigeria MICS6 SPSS used here). A second health band (**36–59 months**, illness/care) from the stakeholder spec is **not** implemented yet.

**Open scientific / stakeholder items:** some **MICS-derived rates** (`school_attendance_rate`, `anc_rate`, `vacc_card_rate`, etc.) remain **predictors** while also relating to deprivation definitions — reviewers flagged **leakage**; removal/replacement with non-survey accessibility layers is a **recommended next step**. **MICS GIS** cluster access for 2021 validation and **multi-age WorldPop** are not wired in.

---

## Grid and supervision

| Item | Detail |
|------|--------|
| Base geometry | **103,424** points — RWI Nigeria grid (**~2.4 km** resolution; other rasters sampled to these centres) |
| Primary targets | MICS6 **composite** moderate / severe prevalence + depth — **37 states** |
| Ridge training signal | **MICS** block + **DHS** nearest-cluster auxiliary block (`dhs_aux_mics_scale: 1.0`, `dhs_aux_dhs_scale: 1.0`) |
| Soft-label blend | Config keys retained; **inactive for training** while DHS auxiliary scale is positive (see `config/config_nga.yaml` `ridge` comments) |
| Population | WorldPop **2020 constrained** raster (`nga_ppp_2020_constrained.tif`) — **under-5** product as used in pipeline docs |

---

## Features (`modeling.features`)

| Block | Count | Notes |
|-------|------:|------|
| Pre-D1 stack | **30** | Geospatial proxies + MICS state×urban/rural aggregates + **2** DHS nearest-cluster columns |
| D1 external | **16** | NBS MPI (9) + NEMIS (7 key columns); state-joined via `subregion` |
| **Total training features** | **46** | Listed in `config/config_nga.yaml` |
| IIAG | **7** columns on `nga_modeling_table.parquet` | **Excluded** from the 46 — national scalars, no within-Nigeria variation for Ridge |

D1 ingestion scripts: `src/scripts/ingest_iiag.py`, `ingest_nbs_mpi.py`, `ingest_nemis.py`, orchestrator `ingest_d1_features.py`. Re-run orchestrator after refreshing raw files, then `python main.py --country nga` to regenerate downstream artifacts.

---

## Models and key outputs

| Model | Role |
|-------|------|
| Ridge | Primary interpretable model; DHS aux-stack; breakdown / themes |
| GBM, GAM, WSNN | Optional; WSNN permutation importance → `nga_wsnn_importance.csv` when WSNN runs |
| Baselines | Uniform, heuristic, RWI redistribution |

**Representative artifacts** (under `Data/outputs/nga/`):

- `tables/nga_predictions.parquet`, `nga_lga_predictions.csv`, `nga_prediction_breakdown.csv`
- `maps/nga_predictions_map.html`, `nga_predictions_map_sample.html`, `nga_comparison_map.html`, `nga_uncertainty_map.html`
- `maps/nga_dimension_comparison_map.html` — seven Kyriaki dimensions + composite panel (when built)
- `eval/evaluation_summary.csv`, `lozo_evaluation.csv`, `dhs_gps_validation.*`, `dhs_aux_stack_sweep.csv`

**Intermediates:** `Data/interim/nga/nga_modeling_table.parquet` (includes D1 + IIAG columns; column count grows with ingestion — **88** columns after D1 merge as of May 2026).

---

## Per-dimension pipeline (Kyriaki specification)

| Dimension | Status |
|-----------|--------|
| Shelter, sanitation, water | Implemented from MICS6 `hh.sav` / `hl.sav` rules in `src/targets/dimension_targets.py` |
| Education 5–14, 15–17 | Implemented from `hl.sav` |
| Health (12–35 months, vaccines) | Implemented from `ch.sav` immunisation variables |
| Nutrition | **MDD proxy** — **HAZ −2 / −3 not available** in the Nigeria MICS6 files used |
| Health (36–59 months, ARI + care-seeking) | **Not implemented** |

Training / prediction helper: `src/scripts/run_dimension_models.py` → `Data/outputs/nga/tables/nga_dimension_predictions.csv`, `eval/nga_dimension_summary.csv`.

---

## Validation snapshot (figures recorded May 4, 2026)

**LOZO (Ridge raw, pre-reconciliation on held-out state):** MAE **11.84** pp; Pearson **0.534**; Spearman **0.463**. Difficult states called out in `PROJECT_STATUS.md` include **Nasarawa**, **Ogun**, **Osun**.

**DHS GPS (1,382 clusters, nearest grid cell):** median distance **~1.0 km**; Ridge Spearman **0.553** with stacked aux at 1.0 (vs **0.486** soft-label baseline in prior sweep notes); Ridge MAE **17.67** pp vs DHS index×100 (cross-metric caveat).

**DHS aux sweep:** `dhs_aux_stack_sweep.csv` — follow-up: rerun **with full LOZO** (not `--skip-lozo`) to confirm held-out state behaviour.

---

## Stakeholder feedback (May 2026 transcript) — alignment

| Topic | Status |
|-------|--------|
| Individual dimension targets vs aggregate-only narrative | **Partially addressed** — dimensions exist alongside **composite** primary model |
| HAZ nutrition | **Blocked by data** — document MDD proxy until HAZ exists |
| Second health age band (36–59m) | **Pending** |
| Grid: 1 km vs RWI ~2.4 km | **Substance = RWI grid**; fix stray **“~1 km²”** wording in `TECHNICAL_OVERVIEW.md` if still present |
| WorldPop age bands | **Pending** — single under-5 raster in config |
| MICS GIS clusters for Nigeria 2021 | **Pending** — not in repo |
| Remove MICS leakage predictors (attendance, ANC, vacc card, …) | **Pending** — still in `config_nga.yaml` |
| HDX accessibility / GHS SMOD2023 / marketplace rasters | **Partial overlap** (Weiss travel time, GHSL SMOD already); **not** the exact HDX / Copernicus 2023 / arxiv marketplace stack as named |

---

## Recommended next steps (priority order)

1. **Leakage fix** — Remove or replace MICS-overlapping predictors; document rationale; rerun `main.py` and eval.  
2. **Docs** — Align `TECHNICAL_OVERVIEW.md` grid wording with §3.1 (~2.4 km).  
3. **MICS GIS** — Obtain cluster geography; optional validation + optional cluster covariates.  
4. **WorldPop** — Add age bands if programme needs “all children” for education-facing targets.  
5. **Dimensions** — Implement **36–59m health** **or** explicitly defer; pursue **HAZ** when data allow.  
6. **Backlog** — **D2** second country; **LOZO-inclusive** aux-stack re-check; **U1** LGA-first comms; optional **GBM SHAP**, **D1 LGA** harmonisation.  
7. **Process** — Update **`PROJECT_STATUS.md`** in the same PR as any of the above.

---

## Git / GitHub — large files (`GH001`)

A push to GitHub was **rejected** when commits contained generated maps and CSVs over **100 MB** (and NBS MPI `.dta` files over that limit). See **`PROJECT_STATUS.md` §4 — “GitHub push limits — GH001”** for the exact file list, limits, and recovery commands. **`.gitignore`** excludes the named full-grid Folium HTML files, the two large prediction CSVs, and all NBS MPI `*.dta` under that survey folder; prefer **Parquet** tables and **`_sample.html`** maps for anything that must live in Git.

---

## Where to read more

| Document | Purpose |
|----------|---------|
| `PROJECT_STATUS.md` | Full checklist, priorities 1–6, completed-since list, master concerns |
| `TECHNICAL_OVERVIEW.md` | Data sources, column reference, architecture |
| `HIGH_LEVEL_MODEL_OVERVIEW.md` | Model narrative, Ridge decomposition, breakdown schema |
| `config/config_nga.yaml` | Paths, `modeling.features`, Ridge DHS knobs, dimensions `feature_overrides` |
| `Data/Nigeria/d1_external/README.md` | D1 raw data layout and re-download notes |

---

*End of snapshot — May 4, 2026.*
