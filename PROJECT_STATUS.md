# Nigeria Child Deprivation Pipeline — Project Status

> **Research tool only. Outputs are NOT official poverty statistics.**
> UNICEF × RBC Borealis AI collaboration.

**Maintaining this file:** After any substantive code, config, data-pipeline, or evaluation change, update this document in the **same change** (do not leave it stale). At minimum: **§1** (feature counts / new datasets), **§6** Master checklist + **Full list of remaining open items** + **Summary** table, and **Completed since last update** when you ship something new. Prefer this file’s checklist over one-line summaries elsewhere (**C8**).

**Dated snapshot (May 4, 2026):** one-page executive status → [`CURRENT_STATUS_MAY_4_2026.md`](CURRENT_STATUS_MAY_4_2026.md). Update or supersede that file when the repo crosses a major milestone.

---

## Table of Contents

1. [What Has Been Built](#1-what-has-been-built)
2. [How the Model Works](#2-how-the-model-works) — including how a cell gets its score and what we can/cannot know
3. [Current Model Performance](#3-current-model-performance) — LOZO full results, permutation test, hierarchical validation
4. [How to Test It](#4-how-to-test-it) — includes **GitHub `GH001` large-file push rejection** (May 2026)
5. [All Outputs](#5-all-outputs)
6. [Next Steps](#6-next-steps) — **remaining TODOs at a glance**, priorities, master checklist, **remaining open items (D2, M3, full backlog)**

---

## 1. What Has Been Built

### Data Infrastructure

| Dataset | Source | What it provides |
|---|---|---|
| RWI (Relative Wealth Index) | Meta / World Bank | 103,424-point base grid across Nigeria |
| Nigeria MICS6 2021 | UNICEF microdata | Ground-truth deprivation targets for 37 states |
| WorldPop 2020 | Population raster | Per-cell child population estimates |
| GHSL SMOD | EU JRC | Urban/rural classification per cell |
| Accessibility rasters | Weiss et al. | Travel time to cities (2 thresholds) |
| VIIRS Nightlights 2019 | NASA | Proxy for economic activity |
| TerraClimate Rainfall 2018 | Climatology Lab | Annual precipitation (mm) |
| ACLED Conflict Events | ACLED | Conflict events + fatalities by area |
| OSM Schools | OpenStreetMap | Distance to nearest school (km) |
| GRID3 / OSM Health Facilities | HDX / OSM | Distance to nearest health facility (km) |
| OSM Building Density | OpenStreetMap | Building count density + log transform |
| GHSL Built-Up Surface 2020 | EU JRC | Satellite building fraction per cell (ghsl_built_frac, log_ghsl_built) |
| LSMS-ISA 2018–2019 | World Bank | Household consumption for sub-state validation |
| Nigeria DHS 2018 (Flat Files + `NGGE7BFL.shp`) | DHS Program | Cluster deprivation + 1,382 geolocated clusters; `merge_dhs_gps` + `validate_predictions_vs_dhs_gps` |
| NBS NLSS 2019 | National Bureau of Statistics Nigeria | State-level monetary poverty headcount (36 states + FCT) — 3rd independent validation source |
| MICS6 Education Indicators | MICS6 hl.sav microdata | School attendance, ever-attended, public school rates — state × urban/rural |
| MICS6 Health Utilization Indicators | MICS6 wm.sav + ch.sav | ANC rate, skilled delivery, facility delivery, vaccination card, diarrhea care — state × urban/rural |
| GADM Admin Boundaries | GADM v4.1 | State (ADM1) + LGA (ADM2) polygons |

**Total features in model: 46** — `modeling.features` in `config/config_nga.yaml`: **30** pre-D1 columns (geospatial + MICS-derived survey aggregates, including **2** DHS nearest-cluster features) + **16** D1 (NBS MPI + NEMIS). See also `HIGH_LEVEL_MODEL_OVERVIEW.md`.

**D1 external datasets now ingested (May 2026):**

| Dataset | Source | Features added |
|---|---|---|
| NBS MPI Microdata (Household survey, ~53k HH) | National Bureau of Statistics Nigeria | 9 state-level features: floor quality, water access, toilet quality, open defecation, food insecurity (HFIAS), health facility distance |
| NEMIS School Listings (4 xlsx: PRE-PRIMARY/PRIMARY/JSS/SSS, ~180k schools) | Federal Ministry of Education / NEMIS | 7 key features: primary school count, enrolment, pupil-per-school ratio, JSS/SSS counts, public % and rural % |
| Mo Ibrahim IIAG 2024 | Mo Ibrahim Foundation | 7 governance scalars stored in modeling table (national-level, excluded from training — enable for multi-country models) |

---

### Models Implemented

| Model | Status | Description |
|---|---|---|
| Ridge Regression | ✅ Full | L2-regularised linear model, cross-validated alpha |
| GBM (LightGBM) | ✅ Full | Gradient boosted trees, feature importance |
| GAM | ✅ Full | Generalised Additive Model (spline-based) |
| WSNN | ✅ Full | Weakly Supervised Neural Network (PyTorch, 3-layer MLP) |
| Uniform Baseline | ✅ | Assigns state average uniformly to all cells |
| Heuristic Baseline | ✅ | Urban/rural split based on SMOD classification |
| RWI Redistribution | ✅ | Inverts RWI to proxy deprivation |

---

### Evaluation Methods Implemented

| Method | What it tests |
|---|---|
| Leave-One-Zone-Out (LOZO) CV | Holds out one state at a time, tests geographic generalization |
| Two-Level CV | Holds out entire states + tests urban/rural split within seen states |
| Hierarchical Cross-Level Validation | Trains on 6 geopolitical zones, evaluates whether 37 state patterns emerge |
| LSMS Household Validation | Compares predictions to actual household consumption data |
| Significance Tests | Paired t-tests between methods |
| RWI Uncertainty Propagation | Tests sensitivity to RWI posterior error |

---

### Output Files Produced

```
Data/outputs/nga/
├── tables/
│   ├── nga_predictions.parquet             # 103,424 grid cells × all models + ridge_theme__* columns
│   ├── nga_predictions.csv                 # same, human-readable
│   ├── nga_prediction_breakdown.csv        # 103,049 rows — per-cell Ridge β·z, theme sums, raw values
│   ├── nga_lga_predictions.csv             # 775 LGAs, pop-weighted aggregates + ridge_bdg__* + raw__*
│   └── nga_full_consolidated.parquet       # features + predictions (one wide table)
├── maps/
│   ├── nga_lga_predictions.geojson         # LGA polygons with deprivation + theme estimates
│   ├── nga_predictions.geojson             # Grid-point predictions (GeoJSON)
│   ├── nga_predictions_map.html            # Interactive Folium map (MarkerCluster, legend, explain popups)
│   ├── nga_predictions_map_sample.html     # Stratified 5 k-cell sample (fast load, demo-safe)
│   ├── nga_uncertainty_map.html            # Interactive CI-width map (legend, dynamic opacity)
│   ├── nga_lga_deprivation_map.png         # Static comparison map
│   └── nga_comparison_map.html            # 6-panel LGA comparison (MICS, Ridge, GAM, error, uncertainty, NBS + theme tooltip)
└── eval/
    ├── evaluation_summary.csv              # All methods × all metrics
    ├── lozo_evaluation.csv                 # LOZO results per state
    ├── hierarchical_validation.csv         # Cross-level validation summary
    ├── hierarchical_validation_detail.csv  # Per-group detail
    ├── two_level_cv.csv                    # Two-level CV results
    ├── gbm_feature_importances.csv         # GBM feature ranking
    ├── significance_tests.csv              # Statistical p-values
    ├── admin_detail_{method}.csv           # Per-state breakdown per model
    ├── dhs_gps_validation.csv|txt          # DHS cluster vs nearest-grid Ridge/RWI
    ├── dhs_soft_label_sweep.csv            # Ridge DHS soft-label weight sweep
    ├── dhs_aux_stack_sweep.csv             # Ridge DHS stacked-aux scale sweep (M1)
    └── ridge_feature_contribution_breakdown.csv  # global Ridge importance / correlation table
```

**New scripts** (not called by `main.py`; run independently):

| Script | Purpose |
|---|---|
| `src/scripts/build_explainability_map.py` | Theme-dominance Folium map (3 layers: dominant theme, direction, DHS contribution) |
| `src/scripts/dhs_aux_sweep.py` | Sweep `dhs_aux_dhs_scale` values, compare DHS Spearman + LOZO MAE to soft-label baseline |
| `src/scripts/build_comparison_map.py` | Rebuild `nga_comparison_map.html` (6-panel LGA map) |

### Key docs in repo

- `PROJECT_STATUS.md` — this file
- `HIGH_LEVEL_MODEL_OVERVIEW.md` — features, data sources, metrics, MICS target definition, Ridge contribution notes
- `Data/README.md` — data layout; DHS raw lives under `Data/Nigeria/dhs/raw/`

---

## 2. How the Model Works

### Architecture Overview

```
Raw Data
   ↓
[Step 01] Build base grid (103,424 cells from RWI CSV)
   ↓
[Step 02] Sample proxy rasters onto grid
          (population, SMOD, travel time, nightlights, building density)
   ↓
[Step 03] Assign each cell to its GADM state (37 states)
   ↓
[Step 04] Compute MICS deprivation targets per state
          (moderate + severe prevalence + depth)
   ↓
[Step 05] Merge features + targets into modeling table
          (new enrichment features: schools, health, conflict, rainfall)
   ↓
[Baselines] Uniform / Heuristic / RWI redistribution
   ↓
[Models] Ridge / GAM / WSNN / GBM → raw predictions per cell
   ↓
[Reconciliation] Population-weighted rescaling so each state
                 exactly matches its MICS official total
   ↓
[Outputs] Grid predictions + LGA aggregates + maps + eval
```

### The Core Idea: Weak Supervision + Reconciliation

The pipeline solves a **disaggregation problem**: MICS gives us one number per state (e.g., "Jigawa has 73% child deprivation"), but we want to know the distribution *within* each state at grid-cell level (~5km).

**Weak supervision** means the model is never trained on ground-truth per-cell labels (none exist). Instead, every model is trained on state-level labels propagated to all cells in that state. The model learns which features (nightlights, travel time, building density, etc.) predict *which cells are worse*, relative to the state average.

**Reconciliation** is the final mandatory step: after prediction, each state's population-weighted mean is algebraically scaled to exactly match the MICS official number. This guarantees that grid cells are consistent with official statistics at state level — but the internal distribution *within* each state comes entirely from the model.

> **What reconciliation means for trust:** You can trust state totals (by construction). You should treat within-state LGA distributions as model estimates that need validation.

---

### How a Cell Gets Its Score — Three Phases

Understanding this precisely matters for interpreting what the outputs mean.

**Phase 1 — Training label (same for every cell in a state)**

The MICS survey gives one number per state. Every cell inside that state is assigned that same number as its training label. There is no cell-level ground truth:

```
All 514 Lagos cells  → labeled 21.75%
All 3,296 Kano cells → labeled 50.05%
All 682 Abia cells   → labeled 17.11%
```

36 unique labels across 103,424 cells.

**Phase 2 — Model predicts a raw score (cells now differ)**

Ridge regression takes the *features* of each cell (RWI, nightlights, building density, travel time, etc.) and predicts a continuous score. Because features differ between cells within the same state, the raw scores now vary:

```
Lagos cells — all officially 21.75%, but raw model output:
cell at lat=6.63, lon=3.26  (RWI=1.45)  →  30.4
cell at lat=6.41, lon=3.28  (RWI=0.66)  →  33.5
cell at lat=6.61, lon=3.94  (RWI=0.89)  →  26.8
```

These raw scores are not calibrated to 21.75%. They encode *relative* spatial variation within the state — which neighbourhoods look more or less deprived compared to each other, according to the proxy features.

**Phase 3 — Reconciliation (force cells to aggregate to official score)**

Each cell's raw score is multiplied by a per-state scale factor:

```
scale_factor = official_target / population_weighted_mean(raw_scores)

Lagos: 21.75 / 30.4 = 0.715
Every Lagos raw score × 0.715 → final reconciled score
```

After reconciliation, the population-weighted average of all cells in a state exactly equals the official MICS figure. This is guaranteed by construction — it is not a model accuracy claim.

```
State    | Official | Pop-wtd reconciled  | Cells
Lagos    |  21.75%  |      21.75%         |   514
Kano     |  50.05%  |      50.05%         | 3,296
Abia     |  17.11%  |      17.11%         |   682
```

---

### What We Know vs. What We Don't

| Claim | Status | Basis |
|---|---|---|
| State aggregates are correct | **Guaranteed** | By construction (reconciliation) |
| Model ranks states correctly (not seen in training) | **Proven** | LOZO r=0.76, permutation p=0.0003 |
| Within-state spatial pattern is plausible | **Partially supported** | LSMS external validation r=0.43 |
| Individual cell scores are accurate | **Unknown — fundamentally unverifiable** | No cell-level ground truth exists or can exist |

**Cell-level ground truth cannot exist.** A "cell" is a ~1 km² square of land. Knowing its true poverty rate would require surveying every household inside it — something never done anywhere in the world at scale. This is not a data gap that will be filled; it is an inherent limit of how surveys work.

What *can* be done is **point-sample validation**: check model predictions at specific locations where household surveys happened (DHS or LSMS clusters). This tests whether the spatial pattern *at surveyed spots* is reasonable, not whether every cell is correct. The LSMS validation (r=0.43) already does this. DHS clusters would extend it to ~1,600–8,000 additional points.

### Features the Model Uses

| Feature | Role |
|---|---|
| `rwi` | Primary wealth signal (Meta/World Bank composite) |
| `population`, `log_population` | Cell-level child count |
| `smod_class`, `is_urban` | Urban/rural/suburban classification |
| `travel_time_cities`, `travel_time_50k` | Market access / isolation |
| `nightlights`, `log_nightlights` | Economic activity proxy |
| `building_density`, `log_building_density` | Physical infrastructure (from OSM, 18.5M buildings) |
| `dist_school_km` | Education access |
| `dist_health_km` | Health system access |
| `conflict_events`, `conflict_fatalities` | Security / fragility proxy |
| `rainfall_mm` | Climate / agricultural productivity |

---

## 3. Current Model Performance

### Are Predictions Better Than Random? Yes — Decisively.

Before looking at the metrics, the fundamental question is whether the spatial structure is real.

**Permutation test (10,000 random label shuffles vs. Ridge):**

| | Pearson r |
|---|---|
| Ridge (actual predictions) | **0.535** |
| Random shuffles (mean across 10,000) | −0.0007 |
| Permutation p-value | **0.0003** |

Only 3 out of 10,000 random assignments achieved a correlation as high as Ridge. The spatial signal is real.

**Significance vs. RWI-only proxy baseline:**

All learned models beat the simple "use Relative Wealth Index as a deprivation proxy" heuristic at p < 10⁻⁴⁰ (Wilcoxon signed-rank test, n=103,424 cells).

---

### LOZO Cross-Validation — What It Measures

LOZO (Leave-One-Zone-Out) is the primary accuracy test. For each state in turn:
1. Remove all cells from that state from the training set
2. Train the model on the remaining 35 states
3. Predict the held-out state using only its features — no reconciliation, no state label used
4. Compare the population-weighted prediction aggregate to the official MICS figure

This tests whether the model can correctly estimate a region it has never seen.

**Summary by model:**

| Model | Mean Abs Error | Pearson r | p-value | Notes |
|---|---|---|---|---|
| **WSNN** | **8.6 pp** | **0.757** | < 0.0001 | Best performer |
| Ridge | 11.6 pp | 0.535 | 0.0008 | Stable, fast, interpretable |
| RWI baseline | 15.5 pp | n/a | — | Simple proxy, no learning |
| Uniform | 15.5 pp | n/a | — | National mean, weakest |
| GAM | 158 pp | 0.052 (n.s.) | 0.76 | **Numerically unstable in LOZO — do not use for generalization** |

**What Pearson r means here:** r=0.757 means that when you pick two random states, the model correctly identifies which one is poorer ~88% of the time, trained only on the other 35 states.

**Full LOZO results (WSNN), every state, sorted by true poverty:**

| State | True % | Predicted % | Error | Notes |
|---|---|---|---|---|
| Abia | 17.1 | 22.8 | +5.6 | |
| Rivers | 19.6 | 15.8 | −3.7 | |
| Lagos | 21.8 | 9.4 | **−12.3** | Outlier — unique mega-city profile |
| Imo | 21.9 | 21.0 | −0.9 ✓ | |
| Anambra | 23.6 | 23.8 | +0.2 ✓ | |
| Kaduna | 25.8 | 47.7 | **+22.0** | Features suggest poverty, survey says not |
| Enugu | 30.6 | 31.0 | +0.4 ✓ | |
| Delta | 31.2 | 29.8 | −1.4 ✓ | |
| Akwa Ibom | 32.8 | 17.7 | −15.2 | |
| Oyo | 34.4 | 39.4 | +4.9 | |
| Ekiti | 35.6 | 25.1 | −10.6 | |
| Borno | 38.9 | 44.8 | +5.9 | |
| Ondo | 39.0 | 34.7 | −4.3 | |
| Ogun | 42.1 | 31.9 | −10.2 | |
| Nasarawa | 47.8 | 88.4 | **+40.6** | Worst miss — rural proxies mislead |
| Kano | 50.1 | 51.2 | +1.1 ✓ | |
| Niger | 52.4 | 53.4 | +1.0 ✓ | |
| Adamawa | 52.6 | 57.9 | +5.2 | |
| Cross River | 53.8 | 32.9 | −20.9 | |
| Plateau | 54.3 | 51.8 | −2.5 ✓ | |
| Bauchi | 59.3 | 66.3 | +7.0 | |
| Katsina | 61.0 | 58.1 | −2.9 ✓ | |
| Gombe | 61.2 | 61.4 | +0.1 ✓ | |
| Benue | 62.2 | 42.1 | −20.1 | |
| Taraba | 63.4 | 67.3 | +4.0 | |
| Zamfara | 64.3 | 62.2 | −2.0 ✓ | |
| Yobe | 65.3 | 66.7 | +1.4 ✓ | |
| Sokoto | 69.9 | 59.3 | −10.6 | |
| Kebbi | 70.1 | 61.3 | −8.7 | |
| Jigawa | 73.1 | 57.9 | −15.2 | |

**Badly predicted states** (error > 20 pp): Nasarawa (+41), Kaduna (+22), Cross River (−21), Benue (−20), Ebonyi (+21). These are states where proxy features tell a misleading story compared to actual household surveys. The model has no way to know this without state-level data.

---

### Hierarchical Validation (trained on 6 zones → predicts 37 states)

The hardest test: train only on 6 geopolitical zones, evaluate on 37 individual states — a level the model was never trained on.

| Experiment | Model | MAE (raw) | Pearson r | Spearman ρ |
|---|---|---|---|---|
| Zone → State | Ridge | 13.3 pp | **0.598** | 0.623 |
| Zone → State | GBM | **11.3 pp** | 0.500 | 0.505 |
| Zone → State × Urban/Rural | Ridge | 18.9 pp | 0.271 | 0.315 |
| Zone → State × Urban/Rural | GBM | 18.6 pp | 0.205 | 0.244 |
| State → State × Urban/Rural | Ridge | 15.7 pp | 0.491 | 0.503 |
| State → State × Urban/Rural | GBM | 14.6 pp | 0.495 | 0.500 |

**Key finding:** r=0.60 from 6 coarse zone labels generalising to 37 states proves the feature set encodes genuine poverty signal. The urban/rural breakdown is weaker (r=0.2–0.3) — sub-state urban/rural splits require finer training supervision.

---

### LSMS Household Validation (external, independent)

- ~4,976 GPS-surveyed households from the LSMS-ISA 2018–2019 survey — not used in any training
- Each household has actual consumption data (income proxy) and a GPS location
- Compared model's grid-cell prediction at each household location to observed consumption
- Pearson r ≈ **0.43** between predicted deprivation rank and inverse consumption rank

This is a point-sample check: at surveyed locations, does the spatial pattern make sense? r=0.43 suggests it does, but with substantial noise. This is the only current evidence that the within-state spatial distribution is partially correct.

---

### GAM Instability — Known Issue

GAM achieves reasonable in-sample fit but has a critical flaw in LOZO: when a high-poverty zone is held out, GAM extrapolates wildly (mean absolute error 158 pp, Pearson r=0.05, not significant). This is a generalisation failure — the GAM overfits the training zones and cannot be trusted for geographic generalization. It should not be used for producing final maps. Ridge and WSNN are the reliable models.

---

### Model Comparison Summary

| Criterion | Best Model | Notes |
|---|---|---|
| Raw spatial accuracy (LOZO) | **WSNN** | r=0.76, MAE=8.6 pp |
| Geographic generalization (hierarchical) | **Ridge** | Most stable; r=0.60 from zone supervision |
| Within-state validation (LSMS) | **Ridge / WSNN** | r≈0.43, similar across models |
| Interpretability | Ridge / GAM | Coefficients + splines readable |
| Uncertainty quantification | Ridge | 90% CI bands; GAM CI bands unreliable |
| Speed | Ridge (~1s) | WSNN ~15s, GAM ~20s |
| Geographic generalization | **Avoid GAM** | Numerically unstable on held-out zones |

---

## 4. How to Test It

### GitHub push limits — `GH001` large files (May 2026)

Pushing branch `franklin-2` to `https://github.com/nargesmn100/BorealisAI_UNICEF.git` was **rejected by GitHub** (`remote: error: GH001: Large files detected`) because several tracked files exceed GitHub’s **100 MB hard limit** (and some exceed the **50 MB** “warning” threshold).

**Files named in the remote hook (representative sizes from the failed push):**

| Path | Approx. size | GitHub |
|------|----------------:|--------|
| `Data/outputs/nga/maps/nga_predictions_map.html` | ~140 MB | **Blocked** (>100 MB) |
| `Data/outputs/nga/maps/nga_uncertainty_map.html` | ~141 MB | **Blocked** |
| `Data/outputs/nga/tables/nga_prediction_breakdown.csv` | ~161 MB | **Blocked** |
| `Nigeria Multidimensional Poverty Index Survey/SECTION D_ DISABILITY FOR 5 YEARS AND ABOVE.dta` | ~245 MB | **Blocked** |
| `Data/outputs/nga/tables/nga_predictions.csv` | ~100 MB | **Warning** (>50 MB) |
| `Nigeria Multidimensional Poverty Index Survey/SECTION C_ ECONOMIC ACTIVITY AND WORK HISTORY.dta` | ~60 MB | **Warning** (>50 MB) |

**Why:** Folium HTML maps embed the full grid payload. Per-cell Ridge breakdown CSVs scale with row count (~100k+). NBS MPI `.dta` sections are survey microdata and can be very large. **Git LFS** is optional for teams that must version binaries; otherwise **do not commit** these artifacts.

**What we did in-repo (May 2026):** `.gitignore` was updated to exclude the **full-grid** Folium HTML names (`nga_predictions_map.html`, `nga_uncertainty_map.html`, `nga_comparison_map.html`, `nga_dimension_comparison_map.html`, `nga_explainability_map.html` if present), the two large **CSV** paths above, and **`Nigeria Multidimensional Poverty Index Survey/**/*.dta`**. Smaller maps (e.g. `*_sample.html`) can remain tracked. Use **`*.parquet`** outputs for tables in Git; keep heavy artifacts **local** or in **LFS** / object storage.

**If large files are already in your last commit(s) but never reached GitHub:** remove them from the index, then recommit (files stay on disk if you only un-track):

```bash
git rm --cached Data/outputs/nga/maps/nga_predictions_map.html Data/outputs/nga/maps/nga_uncertainty_map.html \
  Data/outputs/nga/tables/nga_prediction_breakdown.csv Data/outputs/nga/tables/nga_predictions.csv \
  "Nigeria Multidimensional Poverty Index Survey/SECTION D_ DISABILITY FOR 5 YEARS AND ABOVE.dta" \
  "Nigeria Multidimensional Poverty Index Survey/SECTION C_ ECONOMIC ACTIVITY AND WORK HISTORY.dta"
# …add any other paths `git status` still lists under Data/outputs or the NBS MPI folder…
git commit -m "Stop tracking large outputs and NBS MPI .dta (GitHub GH001)"
```

**If the same blobs already exist in older commits on the branch:** you must **rewrite history** (e.g. [`git filter-repo`](https://github.com/newren/git-filter-repo)) or branch from a clean point before those files were added, then force-push per team policy. GitHub will keep rejecting pushes until no commit in the range contains a file over 100 MB.

---

### Run the full Nigeria pipeline

```bash
# Full pipeline — all models, all evaluation
python main.py --country nga

# Fast run — skip GBM (good for iteration)
python main.py --country nga --skip-gbm

# Models only (no LOZO CV — fastest, saves outputs)
python main.py --country nga --phase models

# Just retrain models (data cached)
python main.py --country nga --skip-gbm --phase all
```

### Inspect prediction outputs

```python
import pandas as pd

# Grid-level predictions (103K cells)
preds = pd.read_parquet("Data/outputs/nga/tables/nga_predictions.parquet")
print(preds[['latitude','longitude','subregion','population',
             'ridge_moderate','wsnn_moderate','moderate_prevalence']].head(20))

# LGA-level aggregates (775 LGAs)
lga = pd.read_csv("Data/outputs/nga/tables/nga_lga_predictions.csv")
print(lga.sort_values('ridge_moderate', ascending=False).head(20))
```

### Open the interactive map

**Cell-level (dense points)** — use a modern desktop browser (files are multi‑MB HTML):

```
Data/outputs/nga/maps/nga_predictions_map.html        ← full 100 k cells (MarkerCluster on)
Data/outputs/nga/maps/nga_predictions_map_sample.html ← ~5 k stratified sample (fast, demo-safe)
```

Each **circle = one modelling grid cell** (point at cell centre). **Colour** shows predicted **moderate deprivation %** using the best-available model column in this order: GBM → GAM → Ridge → RWI. **Click** a marker for popup: subregion/state, RWI, population, **moderate poverty %**, and the Ridge linear **explain** block (top themes + top features).

**What's new (May 2026):** Both maps now include:
- **MarkerCluster** — markers cluster at low zoom levels; click a cluster to expand or zoom.
- **Dynamic opacity/radius** — automatically reduced for high cell counts to reduce overplotting.
- **Floating legend** — "how to read" box (top-left) explains colour scale and popup fields.
- **Explain popups** — Ridge contribution breakdown (top themes + features) shown per cell.

**Interpretation caveat:** At **whole-country zoom**, 10⁵ markers will still overlap even with clustering; that is expected. Use the **sample map** (`_sample.html`) or the **LGA polygon map** below for national-level presentations. The full map is best used zoomed into one state.

**LGA polygons (recommended for national overview)**

```
Data/outputs/nga/maps/nga_comparison_map.html
```

Six layers (MICS, Ridge, GAM, error, uncertainty, NBS). Use layer control — hover polygons instead of inspecting every cell.

**GeoJSON**

```
Data/outputs/nga/maps/nga_lga_predictions.geojson
```

Colour in QGIS / Kepler by `ridge_moderate`; compare to `mics_state_truth`.

### Kepler.gl recipe (GPU-accelerated; handles 100 k+ cells without lag)

1. Open [kepler.gl/demo](https://kepler.gl/demo) in Chrome (or run locally: `pip install keplergl`).
2. Drag `Data/outputs/nga/tables/nga_predictions.parquet` onto the browser canvas  
   *(Parquet is natively supported since Kepler v2.5; alternatively use the CSV).*
3. **Add layer → Point** → set lat/lon to `latitude` / `longitude`.
4. **Colour by:** `ridge_moderate` — use a Sequential (yellow-orange-red) scale.  
   To compare: duplicate the layer, colour second layer by `mics_state_truth`.
5. **Filter** by `subregion` (state) to drill into one state at a time.
6. **Tooltip** fields: `subregion`, `ridge_moderate`, `rwi`, `population`, `ridge_theme__wealth`, `ridge_theme__health_mics` (if breakdown was merged).
7. Export → PNG or HTML snapshot for reports.

For the breakdown layer: use `nga_prediction_breakdown.csv` (same workflow, colour by `ridge_theme__*` columns to see theme dominance).

### Predictions-map UX — concerns & remediation backlog

| Concern | What users see | Why |
| :--- | :--- | :--- |
| **Overplotting** | No fine structure nationally; tiring to explore | Same-size circles densely mask each other |
| **“Solid” / black appearance** | Viewport lacks readable geography | Stacked translucent markers dominate or hide basemap; large inline HTML markers; screenshots don’t reproduce tiles |
| **Equal visual weight per cell** | Hard to spotlight populous hotspots | Fixed radius ignores population |

**Suggested next steps (solve in layers):**

1. **Practice & comms:** Default stakeholder map → **LGA GeoJSON/comparison**, not cell Folium unless zoomed QA.
2. **Code (`main.py` + YAML):** Optional `MarkerCluster` / FastMarkerCluster; reduce `fill_opacity` above `N_cells` threshold; optionally scale `radius` by `√population` (bounded).
3. **Second export:** Coarse **heat/hex aggregation** Folium layer, or capped **sample cells** parquet → lighter demo HTML.
4. **Heavy analysis:** Ship `nga_predictions.parquet` → Kepler.gl for filters/GPU rendering.

*Full ID’d list of concerns and implementation TODOs: **§6 — Master checklist** (C1–C3, U1–U6).*
### Run evaluation reports

```python
import pandas as pd

# Overall method comparison
eval_df = pd.read_csv("Data/outputs/nga/eval/evaluation_summary.csv", index_col=0)
print(eval_df[['admin_mae_mean_pp','pearson_r_vs_neg_rwi','spearman_r_vs_neg_rwi']].sort_values('pearson_r_vs_neg_rwi', ascending=False))

# LOZO by state and method
lozo = pd.read_csv("Data/outputs/nga/eval/lozo_evaluation.csv")
pivot = lozo.pivot_table(index='zone', columns='method', values='abs_error')
print(pivot.sort_values('wsnn'))

# Feature importances (GBM)
fi = pd.read_csv("Data/outputs/nga/eval/gbm_feature_importances.csv")
print(fi.sort_values('importance', ascending=False))
```

### Run the test suite

```bash
pytest tests/ -v
```

### Re-run from scratch (force recompute)

```bash
python main.py --country nga --force-rerun --skip-gbm
```

---

## 5. All Outputs

### Prediction Columns in `nga_predictions.parquet`

| Column | Description |
|---|---|
| `moderate_prevalence` | MICS state-level truth (% moderate deprivation) |
| `ridge_moderate` | Ridge prediction, reconciled |
| `ridge_moderate_lower/upper` | Ridge 90% CI bounds |
| `gam_moderate` | GAM prediction, reconciled |
| `wsnn_moderate` | WSNN prediction, reconciled |
| `gbm_moderate` | GBM prediction, reconciled |
| `rwi_moderate` | RWI baseline |
| `*_severe` | Same columns for severe deprivation threshold |
| `*_depth` | Depth metric (intensity, not just headcount) |
| `ridge_bdg_popup` | Pre-formatted HTML explain snippet (top themes + features) for Folium popups |
| `ridge_theme__wealth` | Ridge β·z contribution sum for wealth theme |
| `ridge_theme__urban_built` | Ridge β·z contribution sum for urban/built-environment theme |
| `ridge_theme__access_services` | Ridge β·z contribution sum for service-access theme |
| `ridge_theme__health_mics` | Ridge β·z contribution sum for MICS health-utilization theme |
| `ridge_theme__edu_mics` | Ridge β·z contribution sum for MICS education theme |
| `ridge_theme__climate_conflict` | Ridge β·z contribution sum for climate/conflict theme |
| `ridge_theme__dhs_cluster` | Ridge β·z contribution sum for DHS nearest-cluster theme |

### LGA Table (`nga_lga_predictions.csv`) — 775 rows

| Column | Description |
|---|---|
| `state` | GADM state name |
| `lga_name` | LGA name (GADM ADM2) |
| `total_population` | Summed child population |
| `pct_urban` | % of LGA cells classified urban |
| `mics_state_truth` | State-level MICS truth (same for all LGAs in a state) |
| `ridge_moderate` | Population-weighted mean Ridge prediction |
| `wsnn_moderate` | Population-weighted mean WSNN prediction |
| `gam_moderate` | Population-weighted mean GAM prediction |
| `ridge_bdg__<feature>` | Pop-weighted mean per-feature Ridge β·z contribution (one column per feature) |
| `raw__<feature>` | Pop-weighted mean raw feature value per LGA (one column per feature) |
| `ridge_theme__*` | Pop-weighted mean theme contribution sums (7 themes; also in GeoJSON) |

---

## 6. Next Steps

### Remaining TODOs (at a glance) — May 2026

| Priority | ID | What is left |
| :---: | :--- | :--- |
| 1 | **D2** | Second-country smoke test — new `config_{iso}.yaml`, MICS + RWI + GADM + rasters, variable-mapping QA. |
| 2 | **Validation** | Re-run `dhs_aux_sweep.py` **with full LOZO** (not `--skip-lozo`) to confirm `dhs_aux_dhs_scale=1.0` on held-out states; refresh eval CSVs. |
| 3 | **U1** | Stakeholder adoption: default narrative = **LGA** maps / GeoJSON, not 100k-cell Folium (process / comms). |
| 4 | **Optional** | **E5 GBM:** enable `export_gbm_shap`, document SHAP for stakeholders. **D1+:** LGA-level NEMIS/NBS harmonisation to GADM ADM2 for sub-state cell features (state join is done). **Nutrition:** HAZ-based target if a round with anthropometry is used. |
| — | **M3** | Asymmetric LOZO loss — **paused** pending product/ethics sign-off. |
| — | **C8** | Keep Summary / §1 counts aligned with this checklist after each release. |

**D1 core ingestion:** ✅ done (`ingest_iiag.py`, `ingest_nbs_mpi.py`, `ingest_nemis.py`, `ingest_d1_features.py`; 16 new training features in `config_nga.yaml`). Re-run `python src/scripts/ingest_d1_features.py --country nga` after refreshing raw D1 files, then `python main.py --country nga` if you want full pipeline outputs regenerated with the new columns.

---

### Current repository status (review snapshot)

- **Nigeria config:** `config/config_nga.yaml` — **46** predictor features in `modeling.features` (**30** pre-D1 + **16** D1; the 30 include **2** DHS nearest-cluster columns); Ridge **DHS stacked auxiliary** active (`dhs_aux_dhs_scale: 1.0`; soft-label blend off when that scale is positive — see `ridge` block in config); GAM **raw-score clipping** enabled; **map UX** flags (`use_folium_cluster`, `folium_cluster_threshold`, `folium_max_cells_full`, `folium_sample_cells`); **explainability** (`export_prediction_breakdown: true`). **IIAG** scalars (7) live in the modeling table for multi-country use but are **not** in the 46-feature training list (no within-Nigeria variation).
- **Pipeline:** `main.py` runs data → baselines → models (Ridge / optional GAM, GBM, WSNN) → eval → outputs. Outputs now include Ridge breakdown CSV, MarkerClustered Folium maps with legends, sampled HTML export, and LGA full-contribution table. Cached intermediates under `Data/interim/nga/`; final artifacts under `Data/outputs/nga/`.
- **DHS raw inputs:** prefer `Data/Nigeria/dhs/raw/` (`NGKR7BFL`, `NGHR7BFL`, `NGGE7BFL`, SPSS bundle); `process_dhs` / `merge_dhs_gps` resolve that path (legacy repo-root folders still work if present).
- **Docs:** `HIGH_LEVEL_MODEL_OVERVIEW.md` (Ridge linear decomposition formula, soft-label vs stacked-aux explanation, breakdown schema), `Data/README.md`; feature contribution table: `Data/outputs/nga/eval/ridge_feature_contribution_breakdown.csv`.
- **Resolved gaps:** FCT/`Fct` harmonization done (`step03_assign_admin.py` `_SUBREGION_HARMONISE`); `pytest` default-config path fixed; all Folium UX improvements implemented; `--force-rerun` from scratch may still require all geospatial zips in config (e.g. `smod_zip`) on the machine.
- **Paused (product decision):** asymmetric loss / “never under-predict” policy for held-out LOZO — not implemented yet.

---

### Master checklist — open concerns & TODOs

Single working list for **concerns** (risks, misunderstood UI) and **TODOs** raised in review, map UX discussion, screenshot feedback, and earlier backlog text. Detail: **§4** (how to read maps), **Priorities 1–6** below, and **§ Remaining open items** (definitions + consolidated backlog).

#### A. Concerns (not all need code)

| ID | Concern | Notes / likely cause |
| :--- | :--- | :--- |
| **C1** | Cell Folium map **unreadable at national zoom** | ~10⁵ overlapping markers. *Mitigated*: MarkerCluster + dynamic opacity + sample map (`_sample.html`) now generated; use LGA map for national overview. |
| **C2** | Viewport looks **solid dark or "black"** (including screenshots) | Usually **overplotting** hiding basemap; less often blocked tiles (offline/VPN), OOM on huge HTML, or `file://` limits. **Mitigation:** use `_sample.html` or **`nga_comparison_map.html`**; serve via `python -m http.server` if needed. |
| **C3** | **No map visible** / empty viewer | Wrong file, incomplete copy, tab crash, or wait for render. Verify file size and pipeline completed `phase_outputs`. |
| **C4** | **Explain block missing** in popups | Stale HTML from before breakdown merge; rerun full pipeline; confirm `nga_prediction_breakdown.csv` exists. |
| **C5** | **GAM** generalisation failure | LOZO can diverge — do not rely on GAM for final national maps without more work (narrative elsewhere). |
| **C6** | **FCT vs `Fct`** | ✅ **Resolved** — `step03_assign_admin.py` `_SUBREGION_HARMONISE` normalises both names. |
| **C7** | **Default `pytest`** without `config/config.yaml` | ✅ **Resolved** — `tests/test_config.py` now falls back to `config_nga.yaml` or any `config_*.yaml`. |
| **C8** | **End-of-section Summary table** may lag code | Prefer this checklist over one-line Summary rows. |

#### B. Modeling / training TODOs

| ID | Task | Status |
| :--- | :--- | :--- |
| **M1** | **DHS aux stack sweep** | ✅ Ran sweep (May 2026): scale=1.0 is best (Spearman 0.553 vs 0.486 for soft-label). Config updated: `dhs_aux_dhs_scale: 1.0` |
| **M2** | Document soft-label vs aux-stack | ✅ YAML inline + `HIGH_LEVEL_MODEL_OVERVIEW.md §3` |
| **M3** | **Asymmetric LOZO loss** | **Paused** (product decision) — see **§ Remaining open items — M3** |
| **M4** | **FCT/`Fct`** harmonization | ✅ `step03_assign_admin.py` `_SUBREGION_HARMONISE`; `compute_mics_deprivation.py` note |

#### C. Explainability / outputs TODOs

| ID | Task | Status |
| :--- | :--- | :--- |
| **E1** | `nga_prediction_breakdown.csv` + merge to **`nga_predictions.*`** | Done |
| **E2** | Ridge explain in **Folium** + **comparison map** theme hints | Done |
| **E3** | **Explainability-first** map (colour by dominant theme, filters) | ✅ `src/scripts/build_explainability_map.py` — 3 Folium layers (theme, direction, DHS) |
| **E4** | **LGA** full per-feature contribution table | ✅ `lga_aggregation.py` — `ridge_bdg__*` + `raw__*` pop-weighted means in CSV |
| **E5** | **SHAP / Permutation importance** GBM/WSNN | ✅ WSNN permutation importance implemented in `weakly_supervised_nn.py` → `nga_wsnn_importance.csv` |
| **E6** | **HIGH_LEVEL_MODEL_OVERVIEW**: Ridge linear vs reconciled | ✅ §8 formula + `nga_prediction_breakdown.csv` schema table added |

#### D. Map / UI TODOs

| ID | Task | Status |
| :--- | :--- | :--- |
| **U1** | Process: default **stakeholder** view = **LGA** comparison / GeoJSON | **Open** — process / comms adoption (not a code blocker) |
| **U2** | `main.py`: optional **MarkerCluster** (YAML flag) | ✅ Done |
| **U3** | `main.py`: **opacity / radius** rules for high cell counts | ✅ Done |
| **U4** | **Sample** or **hex-binned** Folium export | ✅ Done |
| **U5** | Floating **“how to read”** legend on Folium | ✅ Done |
| **U6** | **Kepler.gl** minimal recipe for `nga_predictions.parquet` | ✅ Done |

#### E. Data / expansion TODOs

| ID | Task | Status |
| :--- | :--- | :--- |
| **D1** | EMIS / governance / NBS LGA (Priority 2) | **✅ Implemented (May 4, 2026)** — see §6 and ingestion scripts |
| **D2** | Second country smoke test (Priority 4) | **Open** — template ready; see **§ Remaining open items — D2** |

---

### Remaining open items — what they mean (D1, D2, M3) + full backlog

This subsection ties the **master checklist IDs** to plain-language explanations so stakeholders know *why* something is open and *what* would unblock it.

#### **D1 — EMIS / governance / NBS LGA features**

**What it is:** Extra predictors beyond MICS survey aggregates — especially **school-system quality** (EMIS), **governance** proxies, and **official LGA poverty** where available.

- **EMIS (Education Management Information System):** administrative data from the **Federal Ministry of Education** (and state ministries) on schools, enrolment, completion, teacher ratios, infrastructure, etc. The pipeline already uses **MICS-derived** attendance and proximity (`school_attendance_rate`, `dist_school_km`, …). EMIS would add **LGA-level quality and capacity** signals that surveys do not fully capture.
- **Governance:** e.g. Mo Ibrahim Index subnational scores or similar — optional enrichment for “institutional capacity” at subnational scale.
- **NBS LGA poverty:** National Bureau of Statistics **LGA identifiers and poverty estimates** (often on request) — would allow **LGA-level validation** of maps, not only state-level MICS reconciliation.

**Local acquisition log (May 2026):** see **`Data/Nigeria/d1_external/README.md`**. Summary:

| Source | Status | Notes |
| :--- | :--- | :--- |
| **NEMIS** school listings (`PRE-PRIMARY`, `PRIMARY`, `JSS`, `SSS` `.xlsx`) | Downloaded to `Data/Nigeria/d1_external/nemis/` | **State-level ingestion ✅** (`ingest_nemis.py` → `nga_nemis_state.csv`, merged via `subregion`). **Optional follow-up:** LGA string harmonisation to **GADM ADM2** for true LGA-level cell features (not yet wired). Re-download: `curl -skL` if needed (TLS may require `-k`). |
| **World Bank** national series | `governance/worldbank_nga_primary_enrollment.json` | Country-level only; API, no login. |
| **Mo Ibrahim IIAG** | **On repo:** `2024-IIAG-scores.xlsx` | **Country × year** governance scores (not LGA). Correct IIAG product for D1 “governance” national layer. |
| **NBS MPI microdata** | **On repo:** `Nigeria Multidimensional Poverty Index Survey/*.dta` | **State-level ingestion ✅** (`ingest_nbs_mpi.py` → weighted prevalences by `a1`, joined as `subregion`). **Optional follow-up:** aggregate by `a2` (LGA code) + harmonise to GADM ADM2 for sub-state validation maps. |

Large NEMIS `.xlsx` files are listed in **`.gitignore`** so they stay local unless you remove that rule for Git LFS.

**✅ Fully implemented (May 4, 2026):**
- `src/scripts/ingest_nemis.py` — reads all 4 NEMIS xlsx files, aggregates to state level (count, enrolment, pupil-per-school, public%, rural%), handles missing states with NaN
- `src/scripts/ingest_nbs_mpi.py` — reads Sections A/E/F/I/J from NBS MPI .dta files, computes household-weighted state-level prevalences for 9 WASH/housing/food/health indicators
- `src/scripts/ingest_iiag.py` — extracts 7 governance indicators for Nigeria 2023 from IIAG xlsx
- `src/scripts/ingest_d1_features.py` — orchestrator: calls all three, joins to modeling table, saves `Data/Nigeria/d1_external/nga_d1_features.parquet`
- `config/config_nga.yaml` — 16 new features added under `modeling.features`; per-dimension `feature_overrides` updated with domain-matched D1 subsets
- NBS features (100% state coverage, 0 NaN) improve WASH dimensions; NEMIS (87% state coverage) improves education models (edu_5_14 R=0.985, edu_15_17 R=0.973)

---

#### **D2 — Second country smoke test**

**What it is:** Prove the pipeline is **config-driven**, not Nigeria-specific: copy `config/config_nga.yaml` → `config/config_{iso}.yaml`, point `paths` at another country’s **RWI, GADM, population raster, MICS SPSS**, adjust bbox and admin layer names, then run `python main.py --country {code}` end-to-end.

**Why it is open:** The **code template and config pattern** are ready; each new country still needs **downloaded inputs** (MICS round may use different variable names — `compute_mics_deprivation.py` has Nigeria vs generic branches) and a short **QA pass** (admin spelling, CRS, missing rasters).

**Blockers:** Time to select a pilot country + obtain files; possible **MICS column mapping** work if not Nigeria-style.

---

#### **M3 — Asymmetric LOZO loss** *(paused)*

**What it is:** Today’s training uses **symmetric** squared error: over- and under-predicting a held-out state are penalised the same. **Asymmetric** loss would penalise **under-prediction** of deprivation more (or more than over-prediction) so that LOZO errors skew toward “safer” over-estimates for policy-facing use cases.

**Why it is paused:** This is a **product and ethics** decision (communicating conservative vs unbiased estimates), not only an engineering change. Implementing it touches the Ridge / CV objective and reporting; it should align with UNICEF guidance on **presenting uncertainty** and **not hiding** underestimation risk.

**Next steps if approved:** specify asymmetric weights or pinball loss quantile, implement in `ridge_model.py` (and eval), re-run LOZO and DHS validation, update narrative in maps and reports.

---

#### Full list of remaining open / follow-up items (May 2026)

| ID / item | Type | Status | Notes |
| :--- | :--- | :--- | :--- |
| **D1** | Data / features | **✅ Done** | NEMIS (33 states × 24 features), NBS MPI (37 states × 9 features), IIAG (7 national scalars) — all merged into modeling table. 16 new features active in `config_nga.yaml`. |
| **D2** | Expansion | **Open** | Second-country smoke test — needs MICS + geo stack for chosen country (see above). |
| **M3** | Modeling / policy | **Paused** | Asymmetric LOZO loss — product decision (see above). |
| **U1** | Process / UX | **Open** | Default stakeholder view = **LGA** maps / GeoJSON (`nga_lga_predictions.*`, `nga_comparison_map.html`) rather than national cell Folium; adoption in decks and SOPs. |
| **C8** | Documentation hygiene | **Open** | Summary tables can lag code; this file’s **Master checklist** and **Full list** below are the source of truth — refresh Summary rows after large releases. |
| **C1–C5** | Concerns | **Mitigated / residual** | Cell-map density, black blob, empty viewer, popup explain, GAM LOZO — mitigations documented in §4 and checklist; GAM still not recommended for primary national map without further work (**C5**). |
| **E5 (GBM)** | Explainability | **Optional follow-up** | **WSNN:** permutation importance implemented → `nga_wsnn_importance.csv`. **GBM:** SHAP sample path exists behind `export_gbm_shap` in config; widen use / document if stakeholders need GBM-specific SHAP at scale. |
| **Nutrition (HAZ)** | Data limitation | **Open** | Nigeria MICS6 SPSS lacks HAZ; dimension nutrition uses **MDD proxy** until anthropometry z-scores are computed or a round with HAZ is used. |
| **DHS aux-stack vs LOZO** | Validation | **Follow-up** | Sweep used `--skip-lozo`; re-run `dhs_aux_sweep.py` **with** LOZO when runtime allows to confirm scale=1.0 does not harm held-out states. |
| **Priority 6** | Narrative / future | **Incremental** | DHS GPS cluster features are already in the pipeline; “full point-level training paradigm” remains a **roadmap** item for heavier cluster-level supervision (see Priority 6 section). |

### ✅ Completed since last update (May 3–4, 2026)

| Item | Status |
|---|---|
| GHSL built-up surface (satellite building density) | ✅ Downloaded, processed, in model |
| GBM OpenMP threading crash in hierarchical CV | ✅ Fixed — GBM now runs all 3 hierarchical experiments |
| Config file `config_nga.yaml` | ✅ Reconstructed and committed |
| Nigeria DHS 2018 flat files processed | ✅ 30,713 children, 1,389 clusters, zone-level deprivation computed (`src/scripts/process_dhs.py`) |
| DHS vs MICS cross-validation | ✅ Pearson r=0.96, Spearman ρ=0.77 — strong zone-level agreement; North West flagged |
| NBS NLSS 2019 state poverty data | ✅ Downloaded, processed, 3rd validation source (`Data/Nigeria/nbs/`) |
| NBS vs MICS cross-validation | ✅ Spearman ρ=0.64 — moderate agreement (different poverty concepts) |
| MICS6 school attendance features | ✅ 37 states × urban/rural extracted from hl.sav — 3 new model features |
| MICS6 health utilization features | ✅ 37 states × urban/rural extracted from wm.sav + ch.sav — 5 new model features |
| Pipeline rebuilt with 28+2 features | ✅ Modeling table: base 28 + 2 DHS nearest-cluster; 103,424/103,424 cells |
| Interactive 6-panel comparison map | ✅ `Data/outputs/nga/maps/nga_comparison_map.html` — MICS truth, Ridge, GAM, error, uncertainty, NBS |
| DHS nearest-cluster engineered features | ✅ Added to modeling table (`dhs_nearest_dep_index`, `dist_km_nearest_dhs_cluster`) and included in `config_nga.yaml` |
| Ridge DHS soft-label sweep | ✅ Tested weights 0.1/0.2/0.3/0.4; best external fit at `dhs_soft_label_weight=0.4` (Spearman ρ=0.600, MAE=14.45 pp vs DHS index×100) |
| DHS + LSMS raw folders organized | ✅ `Data/Nigeria/dhs/raw/`; scripts updated to find flat files and shapefile |
| High-level + data layout docs | ✅ `HIGH_LEVEL_MODEL_OVERVIEW.md`, `Data/README.md` |
| Ridge global contribution / correlation table | ✅ `Data/outputs/nga/eval/ridge_feature_contribution_breakdown.csv` |
| Ridge per-cell breakdown CSV + Folium popups + `ridge_theme__*` LGA merge + comparison-map tooltips | ✅ `prediction_breakdown.py`, `main.py`, `lga_aggregation.py`, `build_comparison_map.py` |
| DHS stacked auxiliary Ridge training option | ✅ `dhs_aux_mics_scale` / `dhs_aux_dhs_scale` in `ridge_model.py` + `config_nga.yaml` (tuning TBD — see **Master checklist M1**) |
| FCT/`Fct` admin label harmonization | ✅ `step03_assign_admin.py` `_SUBREGION_HARMONISE`; all cells now matched |
| `pytest` default-config path fix | ✅ `tests/test_config.py` falls back to `config_nga.yaml` or any `config_*.yaml` |
| **Folium MarkerCluster** (YAML flag `use_folium_cluster`) | ✅ `main.py` + `config_nga.yaml` — clusters at low zoom, configurable threshold |
| **Folium dynamic opacity / radius scaling** | ✅ Auto-lowers `fill_opacity` and marker `radius` when `n_cells > folium_cluster_threshold` |
| **Folium stratified sample export** (`nga_predictions_map_sample.html`) | ✅ ~5 k cells stratified by state; fast-load demo-safe map |
| **Floating "how-to-read" legend** on Folium maps | ✅ Both predictions and uncertainty maps now include a legend pane |
| **Explainability-first map** (`build_explainability_map.py`) | ✅ 3 Folium layers: dominant Ridge theme, contribution direction, DHS cluster contribution |
| **LGA full per-feature contribution table** | ✅ `lga_aggregation.py` writes `ridge_bdg__*` + `raw__*` pop-weighted means to LGA CSV |
| **DHS aux-stack sweep script** (`dhs_aux_sweep.py`) | ✅ Sweeps `dhs_aux_dhs_scale` values, outputs `dhs_aux_stack_sweep.csv` |
| **Kepler.gl recipe** in §4 How-To | ✅ Step-by-step guide for GPU-accelerated map exploration |
| **`HIGH_LEVEL_MODEL_OVERVIEW.md`**: Ridge linear decomposition vs reconciled note | ✅ §8 formula + `nga_prediction_breakdown.csv` schema table added |
| **Full pipeline run verified** (May 3, 2026) | ✅ exit_code=0; all 37 states reconciled; 103,049-row breakdown; 775/775 LGAs matched |
| **Per-dimension targets** (Kyriaki spec, May 4, 2026) | ✅ `src/targets/dimension_targets.py` — 7 dimensions × 37 states from ch.sav + hh.sav + hl.sav |
| **Per-dimension Ridge models** (May 4, 2026) | ✅ `src/scripts/run_dimension_models.py` — reconciled predictions for all 103,424 cells × 7 dimensions |
| **`nga_dimension_targets.csv`** | ✅ 37 states × 22 columns (moderate + severe + n per dimension) |
| **`nga_dimension_predictions.csv`** | ✅ 103,424 rows × 11 columns (7 `{dim}_moderate` prediction columns) |
| **LGA dimension rollup** | ✅ `lga_aggregation.py` merges dimension predictions → `nga_lga_predictions.csv` + GeoJSON (775 LGAs) |
| **7-panel dimension comparison map** | ✅ `src/scripts/build_dimension_map.py` → `nga_dimension_comparison_map.html` (Leaflet, LGA polygons, all 8 panels) |
| **D1 external feature ingestion — NEMIS** (May 4, 2026) | ✅ `src/scripts/ingest_nemis.py` → 33 states × 24 school-system features; aggregated from ~180k school records across 4 levels (PRE/PRIMARY/JSS/SSS) |
| **D1 external feature ingestion — NBS MPI** (May 4, 2026) | ✅ `src/scripts/ingest_nbs_mpi.py` → 37 states × 9 WASH/housing/food/health features; weighted from ~53k households (Sections A, E, F, I, J) |
| **D1 external feature ingestion — IIAG** (May 4, 2026) | ✅ `src/scripts/ingest_iiag.py` → 7 governance scalars for Nigeria 2023; stored in modeling table (use for multi-country extension) |
| **D1 orchestrator + modeling table update** (May 4, 2026) | ✅ `src/scripts/ingest_d1_features.py` → 40 D1 columns merged into `nga_modeling_table.parquet` (103,424 cells × 88 cols); 16 new features added to `config_nga.yaml` |
| **Dimension models retrained with D1 features** (May 4, 2026) | ✅ edu_5_14 R=0.985, edu_15_17 R=0.973, health R=0.816, shelter R=0.706, sanitation R=0.709 with NBS/NEMIS feature overrides |
| **DHS aux-stack sweep run** | ✅ `dhs_aux_sweep.py` → scale=1.0 best; config updated to `dhs_aux_dhs_scale: 1.0` |
| **WSNN permutation importance (E5)** | ✅ `weakly_supervised_nn.py` `_wsnn_permutation_importance()` → `nga_wsnn_importance.csv` on next run |
| **Dimension-specific feature sets** | ✅ Per-dimension feature overrides in `config_nga.yaml` `dimensions.feature_overrides` (13–15 features each) |
| **`PROJECT_STATUS.md` hygiene** (May 4, 2026) | ✅ **Maintaining this file** callout at top; **§6 Remaining TODOs (at a glance)**; repository snapshot + Summary + Priority 2 aligned with 46 features and D1 done; DHS config bullets corrected for stacked aux |
| **GitHub `GH001` large-file rejection** (May 2026) | ✅ Documented in **§4** (file table, limits, `git rm --cached` + history note); **`.gitignore`** tightened — named full-grid Folium HTML maps, large prediction CSVs, `Nigeria Multidimensional Poverty Index Survey/**/*.dta` |

---

### Current Metrics Snapshot (latest run — May 4, 2026)


#### Held-out region generalization (LOZO, Ridge, pre-reconciliation on held-out state)

- Mean absolute error (MAE): **11.84 pp** (geopolitical-zone → state cross-validation, Ridge raw)
- Pearson correlation (predicted vs truth, cross-level): **0.534**
- Spearman rank correlation: **0.463**
- Hardest held-out states: **Nasarawa (+40.6 pp overpredict)**, **Ogun (-13.0 pp)**, **Osun (-12.0 pp)**
#### DHS GPS external validation (1,382 clusters, nearest-grid comparison)

- Mean distance DHS point -> nearest grid cell centre: **1.02 km** (median 1.00 km)
- Ridge Spearman ρ: **0.553** (DHS stacked aux at scale=1.0; previously 0.486 with soft-label)
- Ridge MAE: **17.67 pp** (DHS deprivation index ×100 vs Ridge moderate %)
- RWI baseline Spearman ρ: **0.542**
- Note: the scale of DHS dep index differs from MICS moderate %; MAE is cross-metric.

#### Current selected configuration

- `dhs_aux_dhs_scale: 1.0` / `dhs_aux_mics_scale: 1.0` — **stacked DHS auxiliary Ridge training** (see `ridge_model.py`; soft-label blend is inactive for training while the DHS auxiliary scale is positive).
- `use_dhs_soft_label: true` / `dhs_soft_label_weight: 0.4` — kept in config for reference; **not** used for training while stacked aux is on (see `config_nga.yaml` comments).

---

### Priority 1 — DHS 2018 GPS Clusters — **shapefile integrated + feature engineering complete** *(full point-level training loss still TODO)*

**Impact: Highest possible accuracy gain — the single remaining lever.**

#### What DHS actually provides

DHS surveys interview ~20–30 households at specific GPS locations (clusters). Each cluster has a measured deprivation score and a lat/lon coordinate (jittered ±5 km for privacy). Nigeria DHS 2018 has ~1,600 clusters.

#### What DHS does NOT solve

DHS does **not** provide true cell-level ground truth. True cell-level ground truth — knowing the exact poverty rate of every 1 km² square — cannot exist. It would require surveying every household in every cell, which has never been done anywhere. This is a fundamental limit, not a data gap.

#### What DHS actually solves

**1. Better training signal (the main benefit)**

Currently the model trains with 36 state-level labels. Every cell in Lagos is labeled 21.75% regardless of whether it's a slum or a wealthy waterfront neighbourhood.

With DHS clusters, you have ~1,600 training labels at specific locations — each a real household measurement. The model can learn "this cluster near Victoria Island = 8%, this cluster in a dense low-light area = 34%" instead of "all of Lagos = 21.75%."

```
Current:  36 state labels   → 103K cells share 36 numbers
With DHS: ~1,600 cluster labels → model learns from actual place-specific deprivation
```

Expected improvement: LOZO MAE ~8–11 pp → ~4–6 pp; hierarchical r ~0.60 → ~0.75+.

**2. More validation points**

Currently: 4,976 LSMS points for external validation (r≈0.43).
With DHS: +1,600 cluster points to validate spatial pattern at more locations.

This does not prove cell-level accuracy — but it provides more evidence that the spatial pattern is or isn't plausible.

#### Current status after GPS integration

DHS Household and Kids Recode flat files are already processed (`process_dhs.py` done).
Cluster-level deprivation is keyed by `cluster_id` in `Data/Nigeria/dhs/nga_dhs_cluster_deprivation.csv`.

GPS shapefile is read from (first match wins):
```
Data/Nigeria/dhs/raw/NGGE7BFL/NGGE7BFL.shp
# legacy fallbacks: repo root NGGE7BFL/ or Data/Nigeria/dhs/NGGE7BFL.shp
```

Re-run training/evaluation:
```bash
python main.py --country nga --skip-gbm --skip-gam --skip-wsnn
python -m src.scripts.validate_predictions_vs_dhs_gps
```

---

### Priority 2 — School Quality / Governance Features

MICS6 attendance and health utilisation are in the model. **State-level D1 ingestion is done** (May 2026): NEMIS school listings, NBS MPI WASH/housing/food/health survey aggregates, IIAG national scalars on the grid — see **§1** and `src/scripts/ingest_d1_features.py`.

| Dataset | Source | Status |
|---|---|---|
| NEMIS enrolment / school counts (state join) | Federal Ministry of Education | ✅ In `modeling.features` (subset of 16 D1 columns) |
| NBS MPI household survey (state join) | NBS Nigeria | ✅ In `modeling.features` (9 weighted state prevalences) |
| Mo Ibrahim IIAG (national) | Mo Ibrahim Foundation | ✅ In modeling table; **not** in Ridge feature list (constant across Nigeria) |
| EMIS **completion / repetition / class size** by LGA | States / FME | Not in NEMIS xlsx extracts — **future** if tabular product is obtained |
| Subnational governance (state/LGA index) | Various | **Future** — IIAG remains country-level only |
| NBS **official LGA poverty** tables (p-codes aligned to GADM) | NBS Nigeria | **Future** — would unlock LGA-level **validation** of maps (MPI microdata already gives LGA codes `a2` for optional harmonisation work) |

---

### Priority 3 — Poverty Score Breakdown / Explainability

**Done (research pipeline):**

- **`Data/outputs/nga/tables/nga_prediction_breakdown.csv`** — per-cell Ridge β·z, themes, raw values; merged into **`nga_predictions.*`** via `ridge_bdg_popup` + **`ridge_theme__*`** columns.
- **Folium** — cell map popups show Ridge linear **explain** snippet when exported.
- **`nga_comparison_map.html`** — Ridge LGA tooltip appends population-weighted **theme** line where columns exist (`src/scripts/build_comparison_map.py`).
- Ridge global table: **`eval/ridge_feature_contribution_breakdown.csv`** (coef-level, not geographic).

Still **open**:

- ✅ **Explainability-first map** — done: `src/scripts/build_explainability_map.py` (3-layer Folium: dominant theme, direction, DHS contribution).
- ✅ **LGA full per-feature rollup** — done: `lga_aggregation.py` writes `ridge_bdg__*` + `raw__*` pop-weighted means.
- **GBM SHAP** (optional, `export_gbm_shap` in `config_nga.yaml`) — GBM path in `gbm_model.py`; enable when stakeholders need sampled SHAP at cell level.
- **WSNN explainability** — ✅ permutation importance after WSNN fit → `nga_wsnn_importance.csv` (`weakly_supervised_nn.py`).

For **crowded Folium readability**: MarkerCluster, dynamic opacity/radius, sampled export, and floating legend are all now implemented — see **§4** "Predictions-map UX" and **§6 Master checklist** (UI tasks **U2–U6** ✅ all done).

---

### Priority 4 — Generalize to Other Countries

The pipeline is fully config-driven. To add a country (e.g., Albania, Sudan, Ethiopia):

1. Copy `config/config_nga.yaml` → `config/config_{code}.yaml`
2. Download: RWI CSV, GADM boundaries, population raster, MICS microdata
3. Update paths, bounding box, and admin hierarchy in the new config
4. Run:
   ```bash
   python main.py --country {code}
   ```

Countries with MICS6 data available: ~60 countries. RWI covers ~100 low/middle-income countries.

---

### Priority 5 — Per-Dimension Deprivation Targets ✅ Implemented (May 2026)

**Request from Kyriaki (May 3, 2026):** Instead of training on the composite "≥2 of 6 dimensions" aggregate, define and predict each deprivation dimension individually using exact MICS variable definitions.

#### Dimension specifications (Kyriaki)

| Dimension | Age / Unit | MICS source | Moderate threshold | Severe threshold |
|---|---|---|---|---|
| **Shelter** | children < 17 | `hh.sav` (HC3 sleeping rooms + hl.sav household size) | ≥ 3 persons/room | ≥ 5 persons/room |
| **Sanitation** | children < 17 | `hh.sav` WASH (WS11 toilet type, WS14 location) | improved but shared (WS14 = elsewhere) | unimproved / no facility |
| **Water** | children < 17 | `hh.sav` WASH (WS1 source, WS4 travel time) | improved but > 30 min roundtrip | unimproved / surface / no facility |
| **Nutrition** | children < 5 | `ch.sav` (HAZ z-score) | HAZ < −2 | HAZ < −3 |
| **Education 5–14** | children 5–14 | `hl.sav` (ED10A current level, ED4 ever attended) | not currently attending | never attended |
| **Education 15–17** | youth 15–17 | `hl.sav` (ED5A highest level, ED10A current) | not in secondary, no secondary completion | no primary completion |
| **Health** | children 12–35 months | `ch.sav` (IM20/21 Pentavalent, IM26 measles) | missing ≥ 1 of DPT1/2/3 + measles | never vaccinated (IM11 = 2) |

**Nigeria MICS6 data limitation — Nutrition:** HAZ anthropometric z-scores are **not available** in Nigeria MICS6 SPSS files. The Minimum Dietary Diversity (MDD) proxy (food group count from BD8 columns) is used instead. HAZ module would require separate WHO Anthro computation from raw height/weight measurements.

#### What was built

| Artifact | Description |
|---|---|
| `src/targets/dimension_targets.py` | Computes all 7 dimension flags per child/member, aggregates to 37 state-level targets → `nga_dimension_targets.csv` |
| `src/scripts/run_dimension_models.py` | Trains Ridge per dimension, reconciles to state targets, produces `nga_dimension_predictions.csv` + per-dim Folium maps |
| `Data/interim/nga/nga_dimension_targets.csv` | 37 states × 22 columns (moderate + severe + n per dimension) |
| `Data/outputs/nga/tables/nga_dimension_predictions.csv` | 103,424 cells × 11 columns (7 `{dim}_moderate` predictions) |
| `Data/outputs/nga/eval/nga_dimension_summary.csv` | Per-dimension Ridge alpha, train correlation |

#### How to run

```bash
# Compute dimension targets + train all 7 dimension models (maps off for speed)
python src/scripts/run_dimension_models.py --country nga --no-maps

# Run only selected dimensions
python src/scripts/run_dimension_models.py --country nga --dims shelter edu_5_14 health

# Force recompute MICS targets from SPSS (after MICS file update)
python src/scripts/run_dimension_models.py --country nga --recompute-targets
```

#### Dimension prevalence — Nigeria national averages (May 2026 run)

| Dimension | National moderate % | Range across states |
|---|---|---|
| Shelter (overcrowding) | 54.6% | 35.6 – 77.2% |
| Sanitation (improved+shared) | 2.3% | 0.0 – 7.5% |
| Water (improved but far) | 4.0% | 0.4 – 11.1% |
| Nutrition (MDD proxy) | 25.6% | 13.2 – 37.2% |
| Education 5–14 | 23.4% | 3.2 – 65.7% |
| Education 15–17 | 25.5% | 0.7 – 73.9% |
| Health (vaccination) | 87.0% | 73.7 – 97.4% |

---

### Priority 6 — DHS Point-Level Training *(after DHS arrives)*

Once DHS GPS clusters are available, the training paradigm can shift from coarse weak supervision to direct point-level regression:

```
Current:  37 state-level labels  → all 103K cells in each state share one target
With DHS: ~1,600 cluster labels  → each cluster gets a precise local target
          → train on cluster features → predict all 103K cells
          → reconcile to MICS state totals as consistency check
```

Combined with 4,976 LSMS households already processed, this gives ~6,600 real-world training points — the difference between a research prototype and a deployable tool.

---

### Summary

Canonical backlog for anything still open: **§ Remaining TODOs (at a glance)**, **§ Master checklist**, and **§ Remaining open items — definitions & full backlog** (D2, M3, U1, C8, E5-GBM, HAZ, LOZO cross-check, Priority 6). **D1** state-level ingestion is **done**; optional LGA harmonisation remains.

| Next Step | Effort | Impact | Blocker |
|---|---|---|---|
| **DHS GPS join + validation** | ✅ done | High | — |
| DHS **stacked/auxiliary Ridge loss** (`dhs_aux_dhs_scale`) | ✅ Sweep run; **scale=1.0** in `config_nga.yaml` | High | Re-validate with **full LOZO** in sweep (optional follow-up) |
| **Predictions-map UX** (cluster/opacity/sample/legend) | ✅ U2–U6 | **High** stakeholder | — |
| Poverty score breakdown / Folium / comparison / explainability / LGA rollup | ✅ E1–E4, E6 | High | — |
| **WSNN permutation importance** | ✅ E5 (WSNN path) | Medium | Run pipeline **with** WSNN to refresh CSV |
| **GBM SHAP** (optional) | Partial — flag `export_gbm_shap` | Medium | Runtime / dependency |
| FCT/`Fct` + `pytest` config | ✅ done | Medium | — |
| **Per-dimension targets + models + LGA rollup + 7-panel map** | ✅ May 2026 | **Very High** | HAZ not in Nigeria MICS6 (MDD proxy) |
| **D1** NEMIS / NBS MPI / IIAG ingestion | ✅ May 2026 | Medium–High | Optional: **LGA** harmonisation to GADM ADM2; re-run `main.py` to refresh all eval artifacts |
| **D2** Second country smoke test | 1–2 days per country | High | **MICS + geo** for chosen country |
| **M3** Asymmetric LOZO loss | Paused | Low–Med | **Product / ethics** policy |
| **U1** Stakeholder default = LGA view | Adopt in SOPs | Med | Comms / process |
| **C8** Summary vs checklist drift | Ongoing | Low | Editorial discipline |

**Model currently at 46 training features** (28 core geospatial / MICS-proxy columns + 2 DHS nearest-cluster + **16** D1 NBS MPI / NEMIS; **IIAG** stored on the grid but omitted from training as national constants). **Nigeria research pipeline** includes D1 state-level external signals. Suggested next focus: **(1)** **D2** — second country to prove config portability; **(2)** **LOZO-inclusive** aux-stack re-check; **(3)** **U1** adoption; **(4)** optional **GBM SHAP** and **D1 LGA** harmonisation if sub-state validation is required.
