# Nigeria Child Deprivation Pipeline — Project Status

> **Research tool only. Outputs are NOT official poverty statistics.**
> UNICEF × RBC Borealis AI collaboration.

---

## Table of Contents

1. [What Has Been Built](#1-what-has-been-built)
2. [How the Model Works](#2-how-the-model-works)
3. [Current Model Performance](#3-current-model-performance)
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
| GADM Admin Boundaries | GADM v4.1 | State (ADM1) + LGA (ADM2) polygons |

**Total features in model: 20** *(includes GHSL built-up surface added Apr 19, 2026)*

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

### LOZO Cross-Validation (hold out one state, predict it)

| Model | Mean Abs Error | Median Abs Error | Notes |
|---|---|---|---|
| **WSNN** | **8.76 pp** | **7.21 pp** | Best performer |
| Ridge | 11.62 pp | 7.88 pp | Stable, fast |
| RWI baseline | 15.46 pp | 16.98 pp | Strong no-ML baseline |
| Uniform | 15.46 pp | 16.98 pp | Weakest baseline |
| GAM | unstable in LOZO | — | Overfits in some folds |

"pp" = percentage points. A state with true value of 50% is predicted within ~7–11 pp on average.

### Hierarchical Validation (trained on 6 zones → predicts 37 states)

Results with 20-feature model including GHSL. GBM now runs all experiments (threading fix applied Apr 19).

| Experiment | Model | MAE (raw) | Pearson r | What this proves |
|---|---|---|---|---|
| Zone → State | Ridge | 13.3 pp | **0.598** | Model ranks states from features alone, without state supervision |
| Zone → State | GBM | **11.3 pp** | 0.500 | Lower absolute error; GBM now fully operational |
| Zone → State×Urban/Rural | Ridge | 18.9 pp | 0.271 | Urban/rural split weaker without direct training signal |
| State → State×Urban/Rural | Ridge | 15.7 pp | 0.491 | Urban/rural disaggregation from state-level training |
| State → State×Urban/Rural | GBM | 14.7 pp | 0.495 | GBM slightly better on absolute error |

**Key takeaway:** Pearson r ~0.60 when trained on 6 zones and evaluated on 37 states — the model genuinely learns geographic patterns from features, not just memorising labels.

### LSMS Household Validation (external, independent)

- Compared grid-cell predictions to actual household consumption data from the LSMS-ISA 2018–2019 survey
- Pearson r ≈ **0.43** between predicted deprivation rank and inverse consumption rank
- This is an independent external validation not used in training

### Two-Level CV (holds out an entire state)

- Well-predicted states (abs error < 5 pp): Rivers, Plateau, Ogun, Zamfara
- Poorly predicted states (abs error > 20 pp): Kaduna (27 pp), Jigawa (22 pp)
- Mean error on held-out states: **~10.4 pp**

### Model Comparison Summary

| Criterion | Best Model |
|---|---|
| Raw spatial accuracy (LOZO) | WSNN |
| Geographic generalization (hierarchical) | Ridge (most stable) |
| Interpretability / feature effects | GAM |
| Uncertainty quantification | Ridge + GBM (have 90% CI bands) |
| Speed | Ridge (~1 sec), GAM (~20 sec), WSNN (~15 sec) |

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

### ✅ Completed since last update (Apr 19, 2026)

| Item | Status |
|---|---|
| GHSL built-up surface (satellite building density) | ✅ Downloaded, processed, in model — 20 features total |
| GBM OpenMP threading crash in hierarchical CV | ✅ Fixed — GBM now runs all 3 hierarchical experiments |
| Config file `config_nga.yaml` | ✅ Reconstructed and committed |

---

### Priority 1 — DHS 2018 GPS Clusters *(waiting for approval email)*

**Impact: Highest possible accuracy gain — the single remaining lever.**

DHS provides GPS-located survey clusters with individual household microdata. This enables:
- **Point-level training supervision** — replace state-level labels (~37 regions) with cluster-level labels (~1,600 GPS points)
- Expected accuracy improvement: LOZO MAE ~10 pp → ~4–6 pp; LSMS Pearson ~0.43 → ~0.70+
- Enables sub-state validation with actual deprivation data, not just the consumption proxy

When you receive access, place files here:
```
Data/Nigeria/dhs/
├── NGKR7AFL.DTA     # Kids recode (under-5 deprivation indicators)
├── NGHR7AFL.DTA     # Household recode
└── NGGE7AFL.DTA     # GPS cluster coordinates (jittered to 5 km)
```

Then:
```bash
python src/scripts/process_dhs.py      # script to be written on data arrival
python main.py --country nga --force-rerun
```

---

### Priority 2 — School Quality / Governance Features *(data requests needed)*

Currently `dist_school_km` captures proximity but not quality. These features would improve within-state accuracy:

| Dataset | Source | Cost |
|---|---|---|
| Nigeria EMIS (school completion rates) | Federal Ministry of Education Nigeria | Free, requires request |
| Nigeria HMIS (health facility utilization) | NHIA / FMOH | Free, requires request |
| Subnational governance indicators | Mo Ibrahim Foundation | Partially free |
| P-code matched LGA poverty index | NBS Nigeria | Free from NBS website |

---

### Priority 3 — Generalize to Other Countries

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

### Priority 4 — DHS Point-Level Training *(after DHS arrives)*

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
| **DHS GPS clusters** | 1 day implementation | **Very High** | Waiting on approval email |
| School/governance features | 2–3 days | Medium | Data requests to FME / NBS |
| Other countries | 1–2 days per country | High | Country-specific MICS data |
| DHS point-level training | 3–5 days | **Very High** | DHS data arrival |
