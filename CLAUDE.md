# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geospatial ML pipeline that reconstructs grid-level child deprivation patterns from coarse official statistics for Jamaica. Uses weak supervision with hard administrative reconciliation to preserve trusted official totals. The pipeline supports 15 zone×quintile supervision groups (3 zones × 5 wealth quintiles) from MICS data, with the 3 zone-level targets (Urban/Rural/KMA) used for reconciliation. Outputs are research tools, not official statistics.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run full pipeline
python main.py

# Selective execution
python main.py --skip-gbm           # Skip gradient boosting (slower step)
python main.py --force-rerun         # Ignore cached intermediates
python main.py --phase data          # Only data pipeline (steps 01-05)
python main.py --phase baselines     # Data + baselines
python main.py --config path.yaml    # Custom config file

# Run tests
pytest tests/ -v
```

Verification is done via reconciliation checks (diff < 0.001 pp) built into the pipeline, plus a pytest test suite (27 tests).

## Architecture

**Entry point:** `main.py` orchestrates phases: data → baselines → models → evaluation → outputs.

**Configuration:** `config/config.yaml` defines all paths, geospatial settings, model hyperparameters, and feature lists. Paths resolve relative to the auto-detected project root (`src/utils/config_loader.py`).

**Pipeline (`src/pipeline/`):** Steps 01-05 build the modeling table:
1. Load RWI CSV → 1,745-point base grid
2. Sample 5 proxy rasters onto grid (population, SMOD, travel times) + spatial KNN population imputation (912 cells recovered)
3. Spatial join to GADM parish boundaries + Urban/Rural/KMA classification + nearest-parish fallback for 92 coastal cells
4. Extract poverty targets from MICS survey Excel data + quintile targets (15 zone×quintile groups) + sex-disaggregated targets
5. Merge features + targets into single modeling table + soft quintile membership via Gaussian kernel

Intermediate outputs are cached as Parquet in `data/interim/`. Steps reuse cache unless `--force-rerun` is set.

**Models (`src/models/`):** Ridge regression (primary), LightGBM, GAM (optional). All produce per-grid-cell predictions that are then reconciled.

**Reconciliation (`src/reconciliation/admin_reconcile.py`):** Population-weighted rescaling within each admin zone to enforce exact match with official totals. This is non-negotiable — all predictions must pass reconciliation.

**Baselines (`src/baselines/`):** Uniform allocation and RWI-based redistribution. RWI redistribution is a strong baseline that often matches or beats ML models on proxy agreement.

**Evaluation (`src/evaluation/metrics.py`, `src/evaluation/zone_cv.py`):** Compares all methods on metrics computed against proxy indicators. Includes Leave-One-Zone-Out (LOZO) cross-validation for geographic generalization testing, statistical significance tests (paired t-tests), depth metrics (proportion experiencing *any* deprivation), SHAP values for feature importance, and RWI uncertainty propagation analysis. All methods output both headcount (moderate/severe) and depth predictions.

## Key Constraints

- **15 zone×quintile supervision groups** are now available via `step04_prepare_targets.py` (quintile targets) and `step05_merge_features.py` (soft quintile membership). The 3 zone-level targets (Urban/Rural/KMA) are still used for reconciliation, but models can now train on continuous pseudo-targets with finer granularity.
- **The 1,745-point RWI grid is immutable** — all feature engineering happens on this fixed grid.
- **Urban/Rural/KMA classification is deterministic** — SMOD threshold >= 21 for urban, specific parish codes for KMA.
- **10 features:** rwi, population, log_population, smod_class, is_urban, travel_time_cities, travel_time_50k, log_travel_time_cities, log_travel_time_50k, population_imputed (flag indicating imputed values).
- **Missing values:** Population NaN reduced from 912 to 0 via spatial KNN imputation (Agent 3); 92 coastal cells recovered via nearest-parish fallback; ~79-86 cells have missing travel time features.
- **Config flags:** `nearest_parish_fallback`, `population_imputation`, `use_quintile_targets`, `use_rwi_prior`, `use_quintile_target` control data quality and evaluation features.
- **Optional deps:** lightgbm, pygam, and shap are gracefully skipped if not installed.

## Data Layout

- `Data/Geospatial/` — raw input files (rasters, shapefiles, CSVs, Excel)
- `data/interim/` — cached pipeline intermediates (Parquet)
- `data/outputs/tables/` — predictions (CSV + Parquet)
- `data/outputs/maps/` — GeoJSON + interactive Folium HTML maps
- `data/outputs/eval/` — evaluation summaries

## Python Version

Requires Python >= 3.10 (uses `str | None` union type syntax).
