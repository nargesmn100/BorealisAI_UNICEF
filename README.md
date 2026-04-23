# BorealisAI × UNICEF — Fine-Resolution Poverty Mapping
**Weakly Supervised Child Deprivation Disaggregation**

A geospatial ML pipeline that predicts grid-level child deprivation scores at ~1–5 km resolution using only coarse regional survey data as supervision. Built for UNICEF × RBC Borealis AI — Let's SOLVE It, Spring 2026.

> **Disclaimer**: Outputs are **NOT official statistics**. They represent one possible spatial disaggregation consistent with official totals, intended for research and humanitarian prioritization only.

---

## How It Works

Official poverty surveys (MICS, DHS) only report deprivation rates at coarse levels — states, provinces, or zones. This pipeline disaggregates those coarse figures to a fine ~1 km² grid:

1. **Build a grid** from the Relative Wealth Index (Meta/World Bank), giving ~103,000 cells across Nigeria
2. **Attach 20 proxy features** per cell: wealth index, population, nightlights, conflict events, rainfall, distance to schools and health facilities, building density, and more
3. **Train with weak supervision** — the model never sees cell-level labels (none exist). Instead, it learns to distribute deprivation across cells so that population-weighted averages match official survey totals for each region
4. **Reconcile** — predictions are scaled so region totals match official figures exactly
5. **Validate** — hierarchical cross-validation tests whether a model trained on 6 coarse zones can correctly predict patterns in 37 states it never saw during training

**Current countries:** Jamaica (3 supervision zones, ~1,745 cells) · Nigeria (37 states, ~103,000 cells)

---

## For Collaborators — Getting Started

**The processed modeling table is already in the repo.** You can clone and run models within minutes — no raw data downloads required for day-to-day work.

```bash
git clone https://github.com/nargesmn100/BorealisAI_UNICEF.git
cd BorealisAI_UNICEF
pip install -r requirements.txt

# Run models immediately — uses cached data (no downloads needed)
python main.py --country nga --skip-gbm
```

That's it. Steps 01–05 (data pipeline) load from cached Parquet files already in the repo. You'll get full predictions, LOZO cross-validation, hierarchical validation, and output maps in ~15 minutes.

### What's already in the repo

| What | Path | Notes |
|---|---|---|
| Nigeria modeling table | `Data/interim/nga/nga_modeling_table.parquet` | 103,424 cells × 20 features — ready to use |
| Nigeria deprivation targets | `Data/interim/nga/nga_targets*.csv` | MICS6 state + zone + urban/rural targets |
| All feature CSVs | `Data/Nigeria/features/` | ACLED, GRID3 health, schools, LSMS |
| GADM boundaries | `Data/Nigeria/gadm41_NGA.gpkg` | State + LGA polygons |
| RWI base grid | `Data/Nigeria/nga_relative_wealth_index.csv` | 103K grid points |
| Enrichment features | `Data/Nigeria/features/nga_new_features.parquet` | Nightlights, GHSL, rainfall, conflict, schools, health |
| Nigeria config | `config/config_nga.yaml` | All paths and model parameters |
| Previous results | `Data/outputs/nga/eval/*.csv` | Evaluation summaries from last run |

### What is NOT in the repo (raw data files)

These are large binary files excluded by `.gitignore`. You only need them if you want to **regenerate the pipeline from scratch**:

| File | Why excluded | How to get |
|---|---|---|
| `Data/Nigeria/nga_ppp_2020_constrained.tif` | 50 MB raster | Auto-downloads on first run |
| `Data/Nigeria/features/nightlights/viirs_ntl_2019_nga.tif` | 1.2 MB raster | Auto-downloads on first run |
| `Data/Nigeria/features/rainfall/TerraClimate_ppt_2018.nc` | 134 MB NetCDF | Auto-downloads on first run |
| `Data/Nigeria/features/ghsl/*.tif` | 123 MB raster | [EU JRC](https://ghsl.jrc.ec.europa.eu/) — browser download |
| `Data/Nigeria/features/nga_osm_buildings.parquet` | 166 MB | Derived from OSM — re-run `process_ghsl.py` |
| `Data/Nigeria/Nigeria MICS6 SPSS Datasets/*.sav` | Licensed data | [UNICEF MICS](https://mics.unicef.org/surveys) — requires registration |
| `Data/Nigeria/dhs/raw/NGA_2018_GHSP-W4_v03_M_SPSS/*.sav` | Licensed data | [World Bank Microdata](https://microdata.worldbank.org/) |

> **Note:** If you run `--force-rerun` and are missing raw data files, the pipeline will error on that step. Without `--force-rerun`, it loads from cache and skips missing files gracefully.

---

## Running the Pipeline

### Nigeria (main pipeline)

```bash
# Full pipeline — data + all models + evaluation + outputs (~15 min)
python main.py --country nga

# Skip GBM for faster iteration (~8 min)
python main.py --country nga --skip-gbm

# Run only the data phase (build modeling table)
python main.py --country nga --phase data

# Run models only (data already cached)
python main.py --country nga --phase models

# Force re-run all steps even if cached
python main.py --country nga --force-rerun
```

### Jamaica (original prototype)

```bash
# Full Jamaica pipeline
python main.py

# With all model options
python main.py --skip-gbm --skip-gam --skip-wsnn
```

### All CLI flags

| Flag | Description |
|---|---|
| `--country nga` | Run Nigeria pipeline (default: Jamaica) |
| `--phase data` | Run only data pipeline (Steps 01–05) |
| `--phase models` | Run data + baselines + models |
| `--phase eval` | Run through evaluation |
| `--phase all` | Full pipeline (default) |
| `--skip-gbm` | Skip LightGBM (saves ~5 min in LOZO CV) |
| `--skip-gam` | Skip Generalised Additive Model |
| `--skip-wsnn` | Skip Weakly Supervised Neural Network |
| `--force-rerun` | Ignore cache, re-run all steps from scratch |
| `--config PATH` | Use a custom config file |

---

## Required Data *(only needed to regenerate from scratch)*

These files are only needed if you run `--force-rerun` to rebuild the pipeline from raw inputs. For normal use, the cached modeling table in the repo is sufficient (see [For Collaborators](#for-collaborators--getting-started) above).

The pipeline auto-downloads the files marked *(auto)* on first run.

### Nigeria

| File | Path | Source |
|---|---|---|
| Relative Wealth Index | `Data/Nigeria/nga_relative_wealth_index.csv` | [Meta / Humanitarian Data Exchange](https://data.humdata.org/dataset/nigeria-relative-wealth-index) |
| Population raster | `Data/Nigeria/nga_ppp_2020_constrained.tif` | [WorldPop](https://hub.worldpop.org/) *(auto)* |
| GADM boundaries | `Data/Nigeria/gadm41_NGA.gpkg` | [GADM](https://gadm.org/) *(auto)* |
| MICS6 microdata | `Data/Nigeria/Nigeria MICS6 SPSS Datasets/` | [UNICEF MICS](https://mics.unicef.org/surveys) — requires registration |
| VIIRS nightlights | `Data/Nigeria/features/nightlights/viirs_ntl_2019_nga.tif` | [NASA Earthdata](https://eogdata.mines.edu/products/vnl/) |
| GHSL built-up surface | `Data/Nigeria/features/ghsl/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif` | [EU JRC GHSL](https://ghsl.jrc.ec.europa.eu/) |
| ACLED conflict | `Data/Nigeria/features/conflict/acled_nigeria.csv` | [ACLED](https://acleddata.com/) — free registration |
| TerraClimate rainfall | `Data/Nigeria/features/rainfall/TerraClimate_ppt_2018.nc` | [Climatology Lab](https://www.climatologylab.org/terraclimate.html) |
| GRID3 health facilities | `Data/Nigeria/features/health_facilities/grid3_nga_-_health_facilities_-1.csv` | [Humanitarian Data Exchange](https://data.humdata.org/) |
| OSM schools | `Data/Nigeria/features/schools/nga_schools_osm.csv` | Extracted from OpenStreetMap |
| LSMS-ISA 2018 | `Data/Nigeria/lsms/` | [World Bank Microdata](https://microdata.worldbank.org/) |

### Processing GHSL after download

After placing the GHSL TIF file, run this once before the main pipeline:

```bash
python -m src.scripts.process_ghsl
```

---

## Outputs

All outputs are written to `Data/outputs/nga/` (Nigeria) or `Data/outputs/` (Jamaica).

### Prediction tables

| File | Description |
|---|---|
| `tables/nga_predictions.parquet` | 103,424 grid cells × all model predictions + uncertainty bounds |
| `tables/nga_predictions.csv` | Same, human-readable CSV |
| `tables/nga_lga_predictions.csv` | 775 LGAs — population-weighted aggregates of all models |

### Maps

| File | Description |
|---|---|
| `maps/nga_predictions_map.html` | Interactive choropleth map (open in any browser) |
| `maps/nga_uncertainty_map.html` | Interactive uncertainty (90% CI width) map |
| `maps/nga_lga_predictions.geojson` | LGA polygons — load in QGIS or [Kepler.gl](https://kepler.gl/) |
| `maps/nga_predictions.geojson` | Grid-point predictions as GeoJSON |

### Evaluation

| File | Description |
|---|---|
| `eval/evaluation_summary.csv` | All models × all metrics |
| `eval/lozo_evaluation.csv` | Leave-One-Zone-Out results per state |
| `eval/hierarchical_validation.csv` | Cross-level validation (zone → state generalization) |
| `eval/gbm_feature_importances.csv` | LightGBM feature importance ranking |
| `eval/admin_detail_{model}.csv` | Per-state predictions vs truth for each model |

### Inspecting results in Python

```python
import pandas as pd

# Grid-level predictions
preds = pd.read_parquet("Data/outputs/nga/tables/nga_predictions.parquet")
print(preds[["latitude", "longitude", "subregion", "ridge_moderate", "wsnn_moderate"]].head())

# LGA-level aggregates — sorted by most deprived
lga = pd.read_csv("Data/outputs/nga/tables/nga_lga_predictions.csv")
print(lga.sort_values("ridge_moderate", ascending=False).head(20))

# Overall model comparison
summary = pd.read_csv("Data/outputs/nga/eval/evaluation_summary.csv", index_col=0)
print(summary[["admin_mae_mean_pp", "pearson_r_vs_neg_rwi"]].sort_values("pearson_r_vs_neg_rwi", ascending=False))
```

---

## Current Performance (Nigeria)

### Leave-One-Zone-Out Cross-Validation

Each state is held out in turn; models are trained on the remaining 35, then predict the held-out state. No reconciliation applied.

| Model | Mean Abs Error | Notes |
|---|---|---|
| **WSNN** | **8.76 pp** | Best generalization |
| Ridge | 11.62 pp | Most stable, fastest |
| Naive baseline | 15.46 pp | Assigns national mean to all cells |

### Hierarchical Validation — the core scientific test

Trained on **6 geopolitical zones only** → evaluated on **37 states** (never seen during training):

| Model | MAE (raw) | Pearson r | What this proves |
|---|---|---|---|
| Ridge | 13.3 pp | **0.598** | Correctly ranks state deprivation from features alone |
| GBM | 11.3 pp | 0.500 | Lower error but slightly noisier ranking |

Pearson r = 0.60 means the model correctly identifies which states are more deprived than others, trained only on coarse zone-level data — no state-level supervision used.

### External validation (LSMS household data)

Compared grid predictions to actual household consumption from 4,976 GPS-surveyed households (not used in training):

- Pearson r ≈ **0.43** between predicted deprivation rank and inverse consumption rank

---

## Interactive Visualization

The `visualization/` directory contains a standalone Next.js app that explains the pipeline through an interactive spatial demo — useful for presentations, stakeholder briefings, or onboarding new collaborators.

### What it shows

- **Coarse regions view** — 5 stylised administrative regions colored by official poverty score, with region names and survey values labeled directly on the map
- **Fine grid view** — the same territory broken into hundreds of grid cells, each colored by its predicted vulnerability score; hover any cell to see its full feature profile (night lights, building density, accessibility, settlement type, population)
- **Overlay view** — coarse region fills + cell grid simultaneously, with a blue outline on the selected region
- **Aggregation panel** — when a region is selected, shows the formula `Ŷ_region = Σ(ŷᵢ × popᵢ) / Σ(popᵢ)` with real numbers, comparing the population-weighted cell average against the official survey figure
- **Resolution toggle** — switch between coarse / medium / fine grid (~140 / 315 / 560 cells) to simulate different prediction resolutions

### Running it

```bash
cd visualization
npm install       # first time only
npm run dev
```

Then open **http://localhost:3000** in your browser.

> No Python environment, no data downloads. The app uses fully synthetic data generated in-code and runs entirely in the browser.

### Building for production / sharing

```bash
cd visualization
npm run build
npm run start     # serves the optimised build locally
```

Or deploy the `visualization/` folder to any static host (Vercel, Netlify, GitHub Pages via `next export`).

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Repository Structure

```
BorealisAI_UNICEF/
├── config/
│   └── config_nga.yaml          # Nigeria config — paths, features, model params
├── src/
│   ├── pipeline/
│   │   ├── step01_build_grid.py       # RWI CSV → base grid
│   │   ├── step02_sample_proxies.py   # Sample rasters onto grid
│   │   ├── step03_assign_admin.py     # Spatial join to GADM states
│   │   ├── step04_prepare_targets.py  # MICS microdata → state deprivation targets
│   │   ├── step05_merge_features.py   # Merge features + targets → modeling table
│   │   └── run_pipeline.py            # Orchestrate steps 01–05 with caching
│   ├── baselines/
│   │   ├── uniform.py                 # Uniform allocation baseline
│   │   ├── heuristic.py               # Urban/rural split baseline
│   │   └── rwi_redistribution.py      # RWI-based redistribution baseline
│   ├── models/
│   │   ├── ridge_model.py             # Ridge regression + bootstrap uncertainty
│   │   ├── gam_model.py               # GAM with spline terms
│   │   ├── gbm_model.py               # LightGBM / XGBoost + feature importance
│   │   └── weakly_supervised_nn.py    # WSNN — trains via aggregation loss
│   ├── evaluation/
│   │   ├── metrics.py                 # Comparative metrics + significance tests
│   │   ├── zone_cv.py                 # Leave-One-Zone-Out cross-validation
│   │   ├── two_level_cv.py            # Two-level cross-validation
│   │   └── hierarchical_cv.py         # Hierarchical cross-level validation
│   ├── reconciliation/
│   │   └── admin_reconcile.py         # Hard administrative reconciliation
│   ├── outputs/
│   │   └── lga_aggregation.py         # Aggregate grid → LGA level
│   ├── targets/
│   │   └── compute_mics_deprivation.py # MICS SPSS → deprivation targets
│   ├── scripts/
│   │   └── process_ghsl.py            # Process GHSL building density raster
│   └── utils/
│       ├── config_loader.py           # YAML config loading + path resolution
│       ├── admin_mappings.py          # State → geopolitical zone mappings
│       └── geo_utils.py              # Raster sampling, CRS helpers
├── Data/
│   ├── Nigeria/                   # Raw input data (see Required Data above)
│   ├── interim/nga/               # Cached pipeline intermediates (Parquet)
│   └── outputs/nga/               # All predictions, maps, and eval outputs
├── tests/                         # pytest test suite
├── notebooks/
│   └── 01_analysis.ipynb          # Exploratory analysis notebook
├── visualization/                 # Interactive Next.js demo app (see above)
│   ├── app/                       # Next.js app-router pages + layout
│   ├── components/                # PovertyMapSVG, Legend, AggregationPanel, etc.
│   └── lib/data.ts                # Synthetic data generation + color scale
├── main.py                        # Main entry point
├── requirements.txt
└── PROJECT_STATUS.md              # Detailed project status and next steps
```

---

## Adding a New Country

The pipeline is config-driven. To add a country (e.g., Albania):

1. Copy `config/config_nga.yaml` → `config/config_alb.yaml`
2. Update paths, bounding box, and admin hierarchy
3. Download country-specific: RWI CSV, GADM boundaries, population raster, MICS microdata
4. Run: `python main.py --country alb`

Countries with MICS6 data available: ~60 countries. RWI covers ~100 low/middle-income countries.

---

## Team

Spring 2026 cohort — RBC Borealis AI "Let's SOLVE It" program:

- [Narges M. Nezhad](https://www.linkedin.com/in/narges-m/)
- [Franklin Ramirez](https://www.linkedin.com/in/franklin611/)
- [Mara Liwayway David](https://www.linkedin.com/in/maraliwayway/)
- [Krithika Kannan](https://www.linkedin.com/in/krithikakannan06/)
- [Alan Zhou](https://www.linkedin.com/in/alan-zhou-893481246/)

---

## Citation

```
UNICEF × RBC Borealis AI (2026).
Fine-Resolution Poverty Mapping with Coarse-Scale Supervision.
Let's SOLVE It Program, Spring 2026.
```
