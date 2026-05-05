# Nigeria Poverty Prediction — High-Level Overview

This document explains, at a high level:

1. Which data columns are used for prediction
2. Where each column comes from
3. What model is trained (and model type)
4. Current accuracy levels
5. How performance on held-out regions is measured
6. How regions are defined and how subregion predictions are calculated

---

## 1) Data Columns Used for Prediction

The current Nigeria production feature set uses **30 predictor columns**:

1. `rwi`  
2. `population`  
3. `log_population`  
4. `smod_class`  
5. `is_urban`  
6. `travel_time_cities`  
7. `travel_time_50k`  
8. `log_travel_time_cities`  
9. `log_travel_time_50k`  
10. `nightlights`  
11. `log_nightlights`  
12. `building_density`  
13. `log_building_density`  
14. `dist_school_km`  
15. `dist_health_km`  
16. `conflict_events`  
17. `conflict_fatalities`  
18. `rainfall_mm`  
19. `ghsl_built_frac`  
20. `log_ghsl_built`  
21. `school_attendance_rate`  
22. `ever_attended_rate`  
23. `public_school_rate`  
24. `anc_rate`  
25. `skilled_delivery_rate`  
26. `facility_delivery_rate`  
27. `vacc_card_rate`  
28. `diarrhea_care_rate`  
29. `dhs_nearest_dep_index`  
30. `dist_km_nearest_dhs_cluster`

---

## 2) Where Each Data Column Comes From

### Core geospatial + wealth/access columns

- `rwi`: Meta / World Bank Relative Wealth Index point grid  
- `population`, `log_population`: WorldPop raster sampling  
- `smod_class`, `is_urban`: GHSL SMOD urbanicity classes  
- `travel_time_*`, `log_travel_time_*`: accessibility rasters (travel time to cities)

### Built environment + environment + fragility

- `nightlights`, `log_nightlights`: VIIRS nightlights  
- `building_density`, `log_building_density`: OSM buildings-derived density  
- `dist_school_km`: distance to nearest OSM school  
- `dist_health_km`: distance to nearest health facility dataset (GRID3/OSM blend)  
- `conflict_events`, `conflict_fatalities`: ACLED events/fatality features  
- `rainfall_mm`: TerraClimate precipitation  
- `ghsl_built_frac`, `log_ghsl_built`: GHSL built-up surface indicators

### MICS-derived social service columns

From Nigeria MICS6 state x urban/rural aggregates:

- Education: `school_attendance_rate`, `ever_attended_rate`, `public_school_rate`  
- Health utilization: `anc_rate`, `skilled_delivery_rate`, `facility_delivery_rate`, `vacc_card_rate`, `diarrhea_care_rate`

### DHS GPS-derived columns

- `dhs_nearest_dep_index`: deprivation index from nearest DHS geolocated cluster  
- `dist_km_nearest_dhs_cluster`: haversine distance from grid cell to nearest DHS cluster

---

## 3) What Model Is Trained

### Primary model: **Ridge Regression**

- **Type:** Regularized linear regression (`StandardScaler + RidgeCV`, CV over α)
- **Training:** weak supervision (state-level MICS targets as soft labels for each cell)
- **DHS supervision options** (configured in `config_nga.yaml → modeling.ridge`):
  - **Soft-label blend** (current default, `use_dhs_soft_label: true`, `dhs_soft_label_weight: 0.4`):
    cell target = 0.6 × state_target + 0.4 × nearest_DHS_cluster_dep × 100.
  - **Stacked auxiliary loss** (`dhs_aux_dhs_scale > 0`, disables soft-label):
    two-block least-squares — block 1: MICS labels × √(mics_scale); block 2: DHS cluster labels × √(dhs_scale).
    More explicit cluster signal but requires tuning. Run `python src/scripts/dhs_aux_sweep.py --skip-lozo` to sweep scales.

Other implemented models (available in pipeline):

- **GBM** (LightGBM / XGBoost fallback) — nonlinear tree ensemble
- **GAM** — additive spline model *(generalises poorly on LOZO — not for final maps)*
- **WSNN** — weakly supervised neural network

For stable held-out geographic performance, Ridge is currently the most reliable model in active use.

---

## 4) Current Accuracy (Latest Snapshot)

### Held-out region performance (LOZO, Ridge)

- **MAE:** 11.98 percentage points  
- **Pearson (target vs predicted aggregate):** 0.495  
- **Spearman:** 0.636  
- **Held-out regions:** 36 states (one state held out per fold)

### DHS GPS external validation (Ridge)

- **Points:** 1,382 DHS clusters  
- **Mean nearest-grid distance:** 1.02 km  
- **Spearman:** 0.600  
- **Pearson:** 0.584  
- **MAE:** 14.45 pp (DHS deprivation index x100 vs Ridge %)

Interpretation: model has meaningful external spatial signal and moderate geographic generalization, with known hard outlier states.

---

## 5) How It Does on Held-Out Regions

Held-out evaluation uses **LOZO (Leave-One-Zone-Out)** over states:

1. Remove one state from training
2. Train on remaining states
3. Predict cell-level values for the held-out state
4. Aggregate held-out predictions to state level (population-weighted)
5. Compare predicted aggregate vs known MICS state truth

This tests if the model can generalize to unseen geography.

Important: held-out LOZO metrics are evaluated without leaking held-out state truth into fitting for that fold.

---

## 6) Region Structure and How Regional Predictions Are Calculated

## Coarse regions (where we have truth)

- Ground-truth supervision is at **state level (ADM1)** from MICS.
- In this pipeline, the `subregion` training target key corresponds to **state** labels in Nigeria.

## Fine regions (subregions below coarse truth)

Two finer levels are used:

1. **Grid cells** (~103,424 points) — core prediction unit  
2. **LGAs (ADM2)** — 775 units derived by aggregating grid predictions

## How prediction is calculated from coarse to fine

1. Build grid and sample all features per cell  
2. Assign each cell to a state (`subregion`) via spatial join  
3. Train model with state-level deprivation labels  
4. Predict cell-level raw deprivation score  
5. Reconcile cell predictions so each state's population-weighted mean matches official state target  
6. Aggregate reconciled cell predictions to LGA by population-weighted averaging

So state truth constrains totals, while within-state distribution is learned from features.

---

## 7) What This Means for Decision-Making

- **Most trustworthy:** state totals (by design via reconciliation)  
- **Useful for targeting:** within-state ranking patterns at LGA/grid level  
- **Needs continued improvement:** hard held-out outlier states and full DHS point-level loss integration

**Explainability artifacts (implemented):**
- `Data/outputs/nga/tables/nga_prediction_breakdown.csv` — per-cell β·z + theme sums + raw values.
- `Data/outputs/nga/maps/nga_predictions_map.html` — popups include Ridge explain block.
- `Data/outputs/nga/tables/nga_lga_predictions.csv` — LGA-aggregated theme + per-feature sums.
- Run `python src/scripts/build_explainability_map.py` for a theme-dominance map (colours by dominant feature group).

**Open items:** SHAP for GBM/WSNN; dedicated theme-filter dashboard view.

See `§6 Master checklist` in `PROJECT_STATUS.md` for the full list (E3–E6, U-series).

---

## 8) Feature Significance and Contribution (Ridge, latest run)

To quantify "which columns matter most" in the current Ridge model, we computed:

- **Contribution share (%)** = each feature's absolute standardized coefficient divided by the sum across features  
- **Pearson correlation** with final `ridge_moderate` predictions

File:

- `Data/outputs/nga/eval/ridge_feature_contribution_breakdown.csv`

Important interpretation notes:

- These percentages are **model influence shares**, not causal percentages of poverty.
- Correlation here is with model predictions, not direct causality or truth.
- Large shares can reflect scaling/collinearity behavior in linear models.

### Top features by model contribution share

| Feature | Contribution share | Corr with predicted poverty |
|---|---:|---:|
| `rainfall_mm` | 83.58% | -0.510 |
| `building_density` | 7.01% | -0.114 |
| `travel_time_cities` | 4.03% | +0.232 |
| `dist_school_km` | 1.49% | +0.359 |
| `travel_time_50k` | 1.30% | +0.102 |
| `dist_km_nearest_dhs_cluster` | 0.71% | +0.030 |

### DHS-derived feature signal

| Feature | Contribution share | Corr with predicted poverty |
|---|---:|---:|
| `dhs_nearest_dep_index` | 0.0008% | +0.476 |
| `dist_km_nearest_dhs_cluster` | 0.71% | +0.030 |

Interpretation: the nearest DHS deprivation signal has strong positive alignment with final predicted poverty (correlation), while its linear coefficient share is currently small in the global Ridge decomposition.

### How poverty score is calculated (concise formula)

For each grid cell:

1. Build standardized feature vector `z = (x − mean) / std` from the 30 columns (StandardScaler)
2. Compute raw Ridge linear score:
   `raw_score = intercept + Σ_j β_j · z_j`
3. **Reconcile** within each state: multiply all cell scores by a constant so the population-weighted state mean matches the official MICS state target.
4. Output the reconciled score as `ridge_moderate`.

**Important distinction:** the **linear decomposition** (`ridge_bdg__<feature>` = β_j · z_j columns in `nga_prediction_breakdown.csv`) reflects the **pre-reconciliation** linear scores. After reconciliation the relative within-state ordering is preserved but the absolute scale is shifted. This is intentional — the model learns spatial variation; the official total is imposed by reconciliation.

**What `nga_prediction_breakdown.csv` gives you per cell:**

| Column prefix | What it is |
|---|---|
| `ridge_bdg__<feature>` | β_j · z_j contribution for that feature (pre-reconciliation) |
| `ridge_theme__<group>` | Sum of β_j · z_j over features in that theme group |
| `raw__<feature>` | Raw (unstandardized) feature value at that cell |
| `ridge_bdg_linear_pred` | Sum of all β_j · z_j + intercept (= `model.predict(X)`) |
| `ridge_moderate` (in predictions) | Reconciled cell prediction |

The breakdown quantifies "what pushed the score up or down before the state total was imposed." It is the correct decomposition for **within-state ranking** and **feature explanation**.
