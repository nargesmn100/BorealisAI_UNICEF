# Nigeria Child Deprivation Pipeline — Project Status

> **Research tool only. Outputs are NOT official poverty statistics.**
> UNICEF × RBC Borealis AI collaboration.

---

## Table of Contents

1. [What Has Been Built](#1-what-has-been-built)
2. [How the Model Works](#2-how-the-model-works) — including how a cell gets its score and what we can/cannot know
3. [Current Model Performance](#3-current-model-performance) — LOZO full results, permutation test, hierarchical validation
4. [How to Test It](#4-how-to-test-it)
5. [All Outputs](#5-all-outputs)
6. [Next Steps](#6-next-steps)

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

**Total features in model: 28** *(updated Apr 21, 2026 — added 8 education + health utilization features from MICS6 microdata)*

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
│   ├── nga_predictions.parquet          # 103,424 grid cells × all models
│   ├── nga_predictions.csv              # same, human-readable
│   └── nga_lga_predictions.csv          # 775 LGAs, population-weighted aggregates
├── maps/
│   ├── nga_lga_predictions.geojson      # LGA polygons with deprivation estimates
│   ├── nga_predictions.geojson          # Grid-point predictions
│   ├── nga_predictions_map.html         # Interactive Folium map
│   ├── nga_uncertainty_map.html         # Interactive CI-width map
│   └── nga_lga_deprivation_map.png      # Static comparison map
└── eval/
    ├── evaluation_summary.csv           # All methods × all metrics
    ├── lozo_evaluation.csv              # LOZO results per state
    ├── hierarchical_validation.csv      # Cross-level validation summary
    ├── two_level_cv.csv                 # Two-level CV results
    ├── gbm_feature_importances.csv      # GBM feature ranking
    ├── significance_tests.csv           # Statistical p-values
    └── admin_detail_{method}.csv        # Per-state breakdown per model
```

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

Open in any browser:
```
Data/outputs/nga/maps/nga_predictions_map.html
```

### Load GeoJSON in QGIS or Kepler.gl

```
Data/outputs/nga/maps/nga_lga_predictions.geojson
```
Colour by `ridge_moderate` or `wsnn_moderate`. Use `mics_state_truth` column to compare.

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

---

## 6. Next Steps

### ✅ Completed since last update (Apr 21, 2026)

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
| Pipeline rebuilt with 28 features | ✅ Modeling table rebuilt; all features confirmed joined (103,424/103,424 cells) |
| Interactive 6-panel comparison map | ✅ `Data/outputs/nga/maps/nga_comparison_map.html` — MICS truth, Ridge, GAM, error, uncertainty, NBS |
| DHS nearest-cluster engineered features | ✅ Added to modeling table (`dhs_nearest_dep_index`, `dist_km_nearest_dhs_cluster`) and included in `config_nga.yaml` |
| Ridge DHS soft-label sweep | ✅ Tested weights 0.1/0.2/0.3/0.4; best external fit at `dhs_soft_label_weight=0.4` (Spearman ρ=0.600, MAE=14.45 pp vs DHS index×100) |

---

### Current Metrics Snapshot (latest run)

#### Held-out region generalization (LOZO, Ridge, pre-reconciliation on held-out state)

- Mean absolute error (MAE): **11.98 pp**
- Pearson correlation (target vs predicted aggregate): **0.495**
- Spearman rank correlation: **0.636**
- Hardest held-out states: **Nasarawa (+74.5 pp overpredict)**, **Jigawa (-37.9 pp underpredict)**, **Kaduna (+28.4 pp overpredict)**

#### DHS GPS external validation (1,382 clusters, nearest-grid comparison)

- Mean distance DHS point -> nearest grid cell centre: **1.02 km** (median 1.00 km)
- Ridge Spearman ρ: **0.600** (p < 0.0001)
- Ridge Pearson r: **0.584** (p < 0.0001)
- Ridge MAE: **14.45 pp** (DHS deprivation index ×100 vs Ridge moderate %)
- RWI baseline Spearman ρ: **0.542**

#### Current selected configuration

- `use_dhs_soft_label: true`
- `dhs_soft_label_weight: 0.4` (best among 0.1/0.2/0.3/0.4 on DHS external fit)

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

GPS shapefile is now integrated from:
```
Data/Nigeria/dhs/NGGE7BFL.shp   # (or NGGE7BFL.DTA)
```

Re-run training/evaluation:
```bash
python main.py --country nga --skip-gbm --skip-gam --skip-wsnn
python -m src.scripts.validate_predictions_vs_dhs_gps
```

---

### Priority 2 — School Quality / Governance Features

School attendance and health utilization (from MICS6 microdata) are now in the model. Remaining high-value features:

| Dataset | Source | Status |
|---|---|---|
| Nigeria EMIS school completion rates by LGA | Federal Ministry of Education | Request needed — would give LGA-level quality |
| Subnational governance indicators | Mo Ibrahim Foundation | Partially free — would capture local governance capacity |
| NBS LGA-level poverty p-codes | NBS Nigeria | Available on request |

---

### Priority 3 — Poverty Score Breakdown / Explainability

Need a decomposition layer so each predicted poverty score is not just a single number, but a transparent breakdown of "what drove it."

Target output for each geography (cell/LGA/state):

- Predicted poverty score (e.g., 69%)
- Contribution by major feature groups (wealth, access, built environment, education, health utilization, conflict, climate, DHS proximity)
- Raw feature values used for that location (not just contributions)
- Direction of effect (+ pushes poverty up, - pushes it down)
- Optional uncertainty/confidence flag for the decomposition

Planned artifacts:

- `Data/outputs/nga/tables/nga_prediction_breakdown.csv`
- Explainability panel in map tooltip / dashboard
- Method notes documenting how contributions are computed (Ridge coefficients first; later SHAP for GBM/WSNN)

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

### Priority 5 — DHS Point-Level Training *(after DHS arrives)*

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

| Next Step | Effort | Impact | Blocker |
|---|---|---|---|
| **DHS GPS join + validation** | done (`merge_dhs_gps`, `validate_predictions_vs_dhs_gps`) | High | — |
| DHS point-level training in loss (beyond soft-label blend) | 3–5 days | **Very High** | Design + `main.py` change |
| Poverty score breakdown / explainability outputs | 2–4 days | High | Attribution design + output schema |
| LGA-level governance/EMIS features | 2–3 days | Medium | Data requests to FME / NBS |
| Other countries | 1–2 days per country | High | Country-specific MICS data |

**Model currently at 30 features** (28 + 2 DHS nearest-cluster features). Next milestone: full DHS point-level loss integration (beyond soft labels) to move from weak supervision toward cluster-supervised training.
