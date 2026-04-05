# BorealisAI_UNICEF
**AI-Powered Reconstruction of Fine-Scale Child Deprivation for Disaster Forecasting**

A research-first geospatial ML pipeline that reconstructs grid-level patterns of multidimensional child deprivation from coarse official statistics, while preserving trusted administrative totals. Built for UNICEF × RBC Borealis AI — Let's SOLVE It, Spring 2026.

> ⚠️ **Disclaimer**: Outputs are **NOT official statistics**. They represent one possible spatial disaggregation consistent with official totals, intended for research and humanitarian prioritization only.

---

## Introduction

Humanitarian systems need to identify *where* vulnerable children are at fine spatial resolution — especially during climate emergencies. But official child poverty data are only available at coarse administrative levels.

This system bridges that gap by:
1. Using high-resolution proxy data (wealth index, population, settlement type, travel time) to infer within-region deprivation patterns
2. Enforcing consistency with official administrative totals via hard reconciliation
3. Comparing ML approaches against redistribution baselines to determine when models add genuine value

---

## Methods

### Data

| Dataset | Type | CRS | Resolution | Purpose |
|---|---|---|---|---|
| `jam_relative_wealth_index.csv` | Point grid | EPSG:4326 | ~2.3 km | Base grid + RWI feature |
| `jam_pop_2030_CN_100m_R2025A_v1.tif` | Raster | EPSG:4326 | 100 m | Population feature |
| `GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V2_0.zip` | Raster | ESRI:54009 | 1 km | Settlement class feature |
| `cit_017_accessibility_to_cities.zip` | Raster | EPSG:4326 | ~1 km | Travel-time feature |
| `access_50k.zip` | Raster | EPSG:4326 | ~1 km | Travel-time-50k feature |
| `gadm41_JAM.gpkg` | Vector | EPSG:4326 | Parish | Spatial boundary join |
| `ChPov_JAM_CUB.xlsx` | Table | — | Urban/Rural/KMA | Poverty targets |

### Model Architecture

The pipeline has 5 phases:

**Phase 1 — Data Pipeline**: Build a ~1745-point base grid from the Jamaica RWI CSV, sample 5 proxy rasters onto each grid point, spatially assign each point to a GADM parish and Urban/Rural/KMA subregion, and merge with official poverty targets.

**Phase 2 — Baselines**:
- *Uniform*: Assign every cell in a zone the zone-level official prevalence
- *RWI redistribution*: Allocate deprivation inversely proportional to exp(−RWI), then reconcile to zone totals

**Phase 3 — ML Models**:
- *Ridge regression*: Regularized linear model predicting relative deprivation scores from all 9 proxy features
- *Gradient Boosted Trees* (LightGBM): Nonlinear benchmark with native feature importance

**Phase 4 — Reconciliation**: All predictions are population-weighted rescaled within each zone so the zone-level mean exactly matches official targets (verified to `diff < 0.001` pp).

**Phase 5 — Evaluation**: Comparative metrics across all methods.

### Key Design Constraint

> The model predicts **relative spatial allocation patterns**, not absolute official statistics. Hard reconciliation maps predictions back to trusted official totals.

---

## Results

### Evaluation Metrics (Jamaica, 2022 survey, moderate poverty)

| Method | Admin MAE (pp) | Pearson r (vs −RWI) | Spearman r (vs −RWI) |
|---|---|---|---|
| Uniform baseline | ~0.000 | 0.386 | 0.428 |
| RWI redistribution | ~0.000 | **0.932** | **0.972** |
| Ridge regression | ~0.000 | 0.379 | 0.182 |
| Gradient Boosting | ~0.000 | 0.377 | 0.366 |

> **Interpretation**: All methods preserve admin totals exactly. RWI correlation metrics measure proxy agreement — not true deprivation accuracy (see caveats below).

### Administrative Reconciliation

All zone predictions verified at **diff = 0.000000 pp** from official targets:
- Kingston Metropolitan Area (KMA): 34.57%
- Rural: 34.65%
- Urban: 22.92%

### Key Caveats

- We have **only 3 geographic zones** (Urban/Rural/KMA) as target values — models cannot be trained with more degrees of freedom than this
- Correlation with RWI measures proxy agreement, not ground truth reconstruction accuracy
- No fine-resolution ground truth exists for Jamaica — all spatial variation is inferred from proxy signals
- Models may not add value beyond RWI redistribution if the proxy features encode similar signals
- Outputs should not be used as official poverty estimates

---

## Installation & Setup

```bash
# Clone the repo and install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Options
python main.py --skip-gbm          # Skip gradient boosting (faster)
python main.py --force-rerun       # Re-run all steps (ignore cache)
python main.py --phase data        # Only run data pipeline
python main.py --phase baselines   # Data + baselines only
```

**Python >= 3.10 required** (uses `str | None` union type syntax).

---

## Repository Structure

```
BorealisAI_UNICEF/
├── config/
│   └── config.yaml              # All paths, parameters, model settings
├── src/
│   ├── pipeline/
│   │   ├── step01_build_grid.py     # Load RWI CSV → base grid
│   │   ├── step02_sample_proxies.py # Sample rasters onto grid
│   │   ├── step03_assign_admin.py   # Spatial join to parishes + subregions
│   │   ├── step04_prepare_targets.py# Extract Jamaica poverty targets
│   │   ├── step05_merge_features.py # Merge features + targets → modeling table
│   │   └── run_pipeline.py          # Orchestrate steps 01–05
│   ├── baselines/
│   │   ├── uniform.py               # Uniform allocation baseline
│   │   └── rwi_redistribution.py    # RWI-based redistribution baseline
│   ├── models/
│   │   ├── ridge_model.py           # Ridge regression + bootstrap uncertainty
│   │   └── gbm_model.py             # LightGBM / XGBoost model
│   ├── reconciliation/
│   │   └── admin_reconcile.py       # Hard administrative reconciliation
│   ├── evaluation/
│   │   └── metrics.py               # Comparative evaluation metrics
│   └── utils/
│       ├── config_loader.py         # YAML config loading + path resolution
│       └── geo_utils.py             # Raster sampling, CRS helpers, spatial joins
├── data/
│   ├── interim/                 # Cached intermediate outputs (Parquet)
│   └── outputs/
│       ├── tables/              # Prediction tables (CSV + Parquet)
│       ├── maps/                # GeoJSON for visualisation
│       └── eval/                # Evaluation summary + admin detail CSVs
├── Data/Geospatial/             # Raw input data files
├── main.py                      # Main entry point
├── requirements.txt
├── PROJECT-BREAKDOWN.md
├── unicef_rbc_project_understanding.md
└── unicef_rbc_problem_solution_framework.md
```

---

## Future Work / Roadmap

- [ ] Parish-level poverty data (if available) would enable 14-zone reconciliation and much stronger evaluation
- [ ] Additional proxy features: night-time lights, distance to health facilities, school density
- [x] GAM model for smoother, more interpretable spatial surfaces — `src/models/gam_model.py`
- [ ] Cross-country generalization: apply same pipeline to other MICS survey countries
- [ ] SHAP analysis for model interpretability (requires `pip install shap`)
- [x] Interactive map output — Folium HTML saved to `data/outputs/maps/jam_predictions_map.html`
- [ ] Uncertainty propagation through reconciliation step

---

## Team Information

This initiative is part of the Spring 2026 cohort of Borealis AI's "Let's SOLVE It" program. The project team includes:
- [Narges M. Nezhad](https://www.linkedin.com/in/narges-m/)
- [Franklin Ramirez](https://www.linkedin.com/in/franklin611/)
- [Mara Liwayway David](https://www.linkedin.com/in/maraliwayway/)
- [Krithika Kannan](https://www.linkedin.com/in/krithikakannan06/)
- [Alan Zhou](https://www.linkedin.com/in/alan-zhou-893481246/)

---

## Citation

If you use this work in your research, please cite:

```
UNICEF × RBC Borealis AI (2026).
AI-Powered Reconstruction of Fine-Scale Child Deprivation for Disaster Forecasting.
Let's SOLVE It Program, Spring 2026.
```
