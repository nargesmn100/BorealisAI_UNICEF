# Pipeline Phases -- UNICEF x RBC Borealis AI

Jamaica Child Deprivation Reconstruction
Program: RBC Borealis AI -- Let's SOLVE It, Spring 2026

---

## Phase 1 -- Workspace Understanding

Goal: Read all documentation, inspect all datasets, propose folder structure, identify base grid and join strategy.

### Documents Read

- PROJECT-BREAKDOWN.md: high-level objective, disaggregate coarse poverty stats to 2 km grid
- README.md: updated later with full pipeline instructions
- unicef_rbc_problem_solution_framework.md: defines resolution mismatch problem, proposes constraint-aware ML
- unicef_rbc_project_understanding.md: full data inventory, locked design decisions, technical framing

### Datasets Inspected

jam_relative_wealth_index.csv
- Format: CSV point grid
- CRS: EPSG:4326
- Shape: 1,745 rows x 4 columns (latitude, longitude, rwi, error)
- No missing values, ~2.3 km spacing
- CHOSEN AS BASE GRID

jam_pop_2030_CN_100m_R2025A_v1.tif
- Format: GeoTIFF raster
- CRS: EPSG:4326
- Resolution: 100 m (1,360 x 2,879 pixels)
- NoData = -99999

gadm41_JAM.gpkg
- Format: GeoPackage
- CRS: EPSG:4326
- Layers: ADM_0 (country, 1 row), ADM_1 (parishes, 14 rows)
- No ADM_2 available

GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V2_0.zip
- Format: Global GeoTIFF raster (zipped)
- CRS: ESRI:54009 Mollweide
- Resolution: 1 km (18,000 x 36,082 pixels)
- Requires CRS reprojection before sampling

cit_017_accessibility_to_cities.zip
- Format: Global GeoTIFF raster (zipped)
- CRS: EPSG:4326
- Sentinel NoData = -9999
- 79 missing values on the RWI grid after sampling

access_50k.zip
- Format: Global GeoTIFF raster (zipped)
- CRS: EPSG:4326
- Sentinel NoData = -9999
- 86 missing values on the RWI grid after sampling

ChPov_JAM_CUB.xlsx
- Format: Excel
- 355 rows x 25 columns
- Jamaica 2011 + 2022 MICS survey data
- Only 3 geographic zones: Urban, Rural, Kingston Metropolitan Area (KMA)

Child Poverty latest estimates Feb 2026.xlsx
- Format: Excel, global dataset
- Jamaica NOT present -- excluded from pipeline

### Key Decisions

- Base grid: RWI CSV (1,745 points, zero missing values)
- Join key: spatial point-in-polygon join to GADM ADM_1 parishes, then Urban/Rural/KMA labeling
- KMA definition: urban cells in Kingston (JAM.3_1) and Saint Andrew (JAM.6_1) parishes
- Target columns renamed: moderate_prevalence and severe_prevalence
- Modeling frame: learn relative within-zone allocation, enforce zone totals via reconciliation

### Folder Structure Created

```
BorealisAI_UNICEF/
  config/config.yaml
  src/
    pipeline/      (steps 01-05 + orchestrator)
    baselines/     (uniform, rwi_redistribution)
    models/        (ridge_model, gam_model, gbm_model)
    reconciliation/(admin_reconcile)
    evaluation/    (metrics)
    utils/         (config_loader, geo_utils)
  data/
    interim/       (cached Parquet files)
    outputs/
      tables/
      maps/
      eval/
  Data/Geospatial/ (raw input files, unchanged)
  notebooks/       (analysis notebook)
  main.py
  requirements.txt
```

---

## Phase 2 -- Data Pipeline

Goal: Reproducible pipeline producing a single clean modeling table ready for ML.

### Step 01 -- Build Base Grid (step01_build_grid.py)

- Loaded jam_relative_wealth_index.csv
- Dropped rows with missing rwi, latitude, or longitude
- Added cell_id integer index
- Output: data/interim/jam_base_grid.parquet (1,745 rows x 4 columns)

### Step 02 -- Sample Proxy Rasters (step02_sample_proxies.py)

Sampled 4 rasters onto the 1,745 grid points:

population
- Source: jam_pop_2030_CN_100m_R2025A_v1.tif
- NoData (-99999) replaced with 0
- log1p transform applied

smod_class
- Source: GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V2_0.zip
- Mollweide CRS reprojected to WGS84 on the fly

travel_time_cities
- Source: cit_017_accessibility_to_cities.zip
- Sentinel -9999 replaced with NaN, then log1p applied

travel_time_50k
- Source: access_50k.zip
- Sentinel -9999 replaced with NaN, then log1p applied

Technical fix: global rasters caused segmentation faults when loaded into RAM.
Fixed by extracting each ZIP to a temp directory and using rasterio windowed reads.

Output: data/interim/jam_grid_proxies.parquet (1,745 rows x 9 columns)

### Step 03 -- Assign Administrative Regions (step03_assign_admin.py)

- Converted grid points to GeoDataFrame (EPSG:4326)
- Loaded GADM ADM_1 layer (14 parishes) from gadm41_JAM.gpkg
- Spatial point-in-polygon join via geopandas.sjoin
- KMA rule: urban cells in Kingston + Saint Andrew -> KMA subregion
- 92 coastal edge cells fell outside all polygons -> subregion = Unknown, excluded from modeling
- Added is_urban binary flag and smod_label string
- Output: data/interim/jam_grid_admin.parquet (1,745 rows x 14 columns)

### Step 04 -- Prepare Target Variables (step04_prepare_targets.py)

- Loaded ChPov_JAM_CUB.xlsx
- Filtered to Jamaica (ISO code = JAM) and 2022 survey year
- Renamed columns to moderate_prevalence and severe_prevalence
- Extracted 3 rows: Urban, Rural, KMA
- Output: data/interim/jam_targets.csv (3 rows x 5 columns)

### Step 05 -- Merge Features and Targets (step05_merge_features.py)

- Left-joined grid-with-admin to targets on subregion
- Imputation: travel times -> subregion median fallback to global median
- Imputation: population NaN -> 0
- Imputation: smod_class NaN -> subregion mode
- Added in_modeling_sample flag (True for 1,653 of 1,745 cells)
- Output: data/interim/jam_modeling_table.parquet (1,745 rows x 18 columns)

---

## Phase 3 -- Baselines

Goal: Two redistribution baselines, reconciled exactly to official zone totals.

### Baseline 1 -- Uniform Allocation (src/baselines/uniform.py)

- Every cell in a zone gets the zone-level official prevalence unchanged
- Result: zero within-zone spatial variation (flat prediction)
- Columns added: uniform_moderate, uniform_severe
- Reconciliation error: 0.000000 pp

### Baseline 2 -- RWI Redistribution (src/baselines/rwi_redistribution.py)

- Raw score = exp(-rwi) per cell (lower wealth = higher deprivation)
- Scores population-weighted to match official zone total
- Formula: reconciled_i = raw_score_i / weighted_zone_mean x zone_target
- Columns added: rwi_moderate, rwi_severe
- Reconciliation error: 0.000000 pp

---

## Phase 4 -- ML Models

Goal: Interpretable models predicting relative within-zone deprivation, then reconciled.

### Feature Set (9 features)

- rwi: Relative Wealth Index
- population: WorldPop 2030 count
- log_population: log1p of population
- is_urban: binary (1 = SMOD urban)
- smod_class: integer settlement code (10-30)
- travel_time_cities: log travel time to nearest city
- travel_time_50k: log travel time to nearest 50k centre
- log_travel_time_cities: same as above (explicit log column)
- log_travel_time_50k: same as above (explicit log column)

### Model 1 -- Ridge Regression (src/models/ridge_model.py)

Architecture: sklearn RidgeCV with StandardScaler
- Cross-validated alpha selection from [0.01, 0.1, 1.0, 10.0, 100.0]
- One model per target (moderate, severe)
- Uncertainty: 50 bootstrap resamples -> 5th/95th percentile bands
- Columns added: ridge_moderate, ridge_severe
- Uncertainty columns: ridge_moderate_lower, ridge_moderate_upper
- Coefficient table logged for interpretability

### Model 2 -- Generalised Additive Model (src/models/gam_model.py)

Architecture: pygam LinearGAM with GCV lambda selection
- Spline terms s() for continuous features, linear terms l() for binary/class
- RWI spline constrained monotone-decreasing (richer = less deprived)
- Uncertainty: built-in GAM confidence bands (95% pointwise)
- Partial dependence curves available via get_partial_dependence()
- Requires: pip install pygam
- Skip flag: python main.py --skip-gam
- Columns added: gam_moderate, gam_severe, gam_moderate_lower, gam_moderate_upper

### Model 3 -- Gradient Boosted Trees (src/models/gbm_model.py)

Architecture: LightGBM LGBMRegressor (falls back to XGBoost if needed)
- Same 9-feature set as Ridge
- Uncertainty: quantile regression alpha=0.1 and 0.9
- Feature importances saved as ranked DataFrame
- Skip flag: python main.py --skip-gbm
- Columns added: gbm_moderate, gbm_severe, gbm_moderate_lower, gbm_moderate_upper

---

## Phase 5 -- Administrative Reconciliation

Goal: All grid-level predictions sum exactly to official zone totals.

Implementation (src/reconciliation/admin_reconcile.py):

  reconciled_i = raw_pred_i x (zone_official_mean / zone_pop_weighted_pred_mean)

Applied inside every model and baseline runner, not as a separate pass.

Verification results (all methods x all zones):

  uniform_moderate  | KMA   | diff = 0.000000 pp OK
  uniform_moderate  | Rural | diff = 0.000000 pp OK
  uniform_moderate  | Urban | diff = 0.000000 pp OK
  rwi_moderate      | all zones: 0.000000 pp OK
  ridge_moderate    | all zones: 0.000000 pp OK
  gam_moderate      | all zones: 0.000000 pp OK
  gbm_moderate      | all zones: 0.000000 pp OK

Official zone targets -- Jamaica 2022 moderate poverty:

  KMA   = 34.57%
  Rural = 34.65%
  Urban = 22.92%

---

## Phase 6 -- Evaluation

Goal: Compare all methods on accuracy, rank quality, admin consistency, uncertainty.

Module: src/evaluation/metrics.py

Metrics computed:

- Admin MAE: mean abs error of zone weighted mean vs official target
- Pearson r vs -RWI: correlation with RWI signal (proxy agreement, not ground truth)
- Spearman r vs -RWI: rank-order correlation with RWI
- CI width: mean 90% confidence interval width
- Top-K overlap: fraction of top-K cells shared with RWI baseline

Results -- Jamaica 2022, moderate poverty:

  Uniform baseline   | Admin MAE ~0.000 | Pearson r = 0.386 | Spearman r = 0.428
  RWI redistribution | Admin MAE ~0.000 | Pearson r = 0.932 | Spearman r = 0.972
  Ridge regression   | Admin MAE ~0.000 | Pearson r = 0.379 | Spearman r = 0.182
  Gradient Boosting  | Admin MAE ~0.000 | Pearson r = 0.377 | Spearman r = 0.366

IMPORTANT CAVEATS:
1. Admin MAE = 0 for all methods by construction (hard reconciliation).
2. Pearson/Spearman r is measured against -RWI, not real ground truth.
3. RWI redistribution scores highest on r(-RWI) because it IS a function of RWI -- circular.
4. No fine-resolution ground truth exists for Jamaica -- absolute accuracy is unknown.

---

## Phase 7 -- Outputs

Goal: Save clean reusable files for tables, maps, and evaluation.

Files generated:

  data/outputs/tables/jam_predictions.parquet      (all 1,745 rows, all method columns)
  data/outputs/tables/jam_predictions.csv          (same, CSV format)
  data/outputs/maps/jam_predictions.geojson        (1,653 modeling-sample cells, with geometry)
  data/outputs/eval/evaluation_summary.csv         (comparative metrics table)
  data/outputs/eval/admin_detail_uniform_moderate.csv
  data/outputs/eval/admin_detail_rwi_moderate.csv
  data/outputs/eval/admin_detail_ridge_moderate.csv
  data/outputs/eval/admin_detail_gbm_moderate.csv

Intermediate cached files:

  data/interim/jam_base_grid.parquet      (Step 01 output)
  data/interim/jam_grid_proxies.parquet   (Step 02 output)
  data/interim/jam_grid_admin.parquet     (Step 03 output)
  data/interim/jam_targets.csv            (Step 04 output)
  data/interim/jam_modeling_table.parquet (Step 05 output)

---

## Analysis Notebook

File: notebooks/01_analysis.ipynb
Run AFTER python main.py has populated data/interim/ and data/outputs/.

Sections and figures:

  Section 1 -- Exploratory Data Analysis
    Fig 1: grid scatter coloured by subregion + RWI colour map
    Fig 2: feature boxplots by zone + correlation matrix

  Section 2 -- Target Data
    Fig 3: official zone-level moderate and severe poverty bar charts

  Section 3 -- Prediction Comparison
    Fig 4: violin plots of within-zone prediction distributions per method
    Fig 5: pairwise scatter RWI redistribution vs other methods

  Section 4 -- Spatial Maps
    Fig 6: 4-panel scatter maps across Jamaica (shared colour scale)

  Section 5 -- Reconciliation Verification
    Fig 7: grouped bar chart achieved zone means vs official targets

  Section 6 -- Uncertainty Bands
    Fig 8: Ridge bootstrap CI width histogram + spatial map
    Fig 9: per-zone prediction band plots

  Section 7 -- Feature Importance
    Fig 10a: Ridge standardised coefficient bar chart
    Fig 10b: GAM partial dependence curves per feature
    Fig 11: GBM gain-based feature importances

  Section 8 -- Evaluation Summary
    Fig 12: comparative metric bar charts + table

  Section 9 -- Key Findings and Limitations
    Honest narrative of what the pipeline can and cannot claim

How to run the notebook:

  pip install jupyter matplotlib pygam
  jupyter notebook notebooks/01_analysis.ipynb

---

## Bugs Fixed

Bug 1: ModuleNotFoundError: rasterio
  Cause: library not installed
  Fix: pip install rasterio geopandas shapely pyogrio

Bug 2: PermissionError operation not permitted
  Cause: sandbox blocking ZIP extraction
  Fix: re-ran with elevated permissions

Bug 3: KeyError GHS_SMOD tif not in archive
  Cause: config_loader was converting zip-internal filenames to absolute filesystem paths
  Fix: separated zip_contents (internal names) from paths (filesystem paths) in config.yaml

Bug 4: KeyError grid_file
  Cause: YAML indentation error nested interim_dir and grid_file under zip_contents
  Fix: corrected indentation so all path keys are directly under paths:

Bug 5: segmentation fault (exit code 139) sampling SMOD/accessibility rasters
  Cause: loading 18,000 x 36,000 pixel global rasters entirely into RAM
  Fix: extract ZIP to temp dir first; rasterio reads from disk with windowed I/O

Bug 6: RuntimeWarning invalid value in log1p
  Cause: accessibility rasters use -9999 as NoData sentinel
  Fix: replace -9999 with np.nan before applying log1p

Bug 7: KeyError Prevalence moderate child poverty (%)
  Cause: baselines and models used raw Excel column names after step04 had renamed them
  Fix: all scripts updated to use moderate_prevalence and severe_prevalence

---

## How to Run the Pipeline

Install dependencies:

  pip install -r requirements.txt

Full pipeline (Uniform + RWI + Ridge + GAM + GBM):

  python main.py

Individual phases:

  python main.py --phase data        (data pipeline only)
  python main.py --phase baselines   (data + baselines)
  python main.py --phase models      (data + baselines + models)
  python main.py --phase eval        (full pipeline through evaluation)

Options:

  python main.py --skip-gbm          (skip gradient boosting, faster)
  python main.py --skip-gam          (skip GAM, if pygam not installed)
  python main.py --force-rerun       (ignore cached intermediate files)

Notebook (run after pipeline):

  jupyter notebook notebooks/01_analysis.ipynb

Requires Python >= 3.10

---

Last updated: April 2026
