# Nigeria Child Deprivation Pipeline — Technical Overview

> UNICEF × RBC Borealis AI collaboration.
> Research tool only. Outputs are NOT official poverty statistics.

---

## Table of Contents

1. [What Problem We Are Solving](#1-what-problem-we-are-solving)
2. [Why Nigeria and Not Jamaica](#2-why-nigeria-and-not-jamaica)
3. [Data Sources](#3-data-sources)
4. [Consolidated Dataset — Column Reference](#4-consolidated-dataset--column-reference)
5. [How We Train the Model](#5-how-we-train-the-model)
6. [How We Make Predictions](#6-how-we-make-predictions)
7. [How We Validate Accuracy](#7-how-we-validate-accuracy)
8. [Current Results](#8-current-results)
9. [What Is Left](#9-what-is-left)

---

## 1. What Problem We Are Solving

Official surveys (MICS, DHS) measure child poverty at the **state level** — one number for each of Nigeria's 37 states. This is the "coarse ground truth." What we do not know from those surveys alone is how that poverty is **distributed within a state** — whether it is concentrated in rural northern communities, around conflict zones, or spread evenly.

**The goal:** Given one coarse number for a state (e.g. Kano = 50% moderate deprivation), predict a deprivation score for every ~1 km² cell inside that state, such that when you population-weight-average all cells back up, you recover the original 50% exactly.

```
MICS says:      Kano State = 50.05% moderate child deprivation
                ↓
Model predicts: each of Kano's 3,296 grid cells gets its own score
                ↓
Guarantee:      population-weighted average of all cells = 50.05%  ✓
                ↓
Output:         which LGAs, which rural clusters, which urban pockets
                are most deprived within Kano
```

This is called **weakly supervised spatial disaggregation**: the model is supervised by coarse region-level totals, not by direct cell-level measurements.

---

## 2. Why Nigeria and Not Jamaica

The original problem statement (Section 8) specified starting with Jamaica as a faster proof-of-concept. Jamaica was partially implemented: a working pipeline was built, the grid was constructed, RWI and population features were sampled, and baseline predictions were produced. That work is retained in `Data/outputs/` (Jamaica predictions) and `config/` (the Jamaica config is now merged into the Nigeria config architecture).

**Jamaica was set aside for three reasons:**

1. **Scale mismatch.** Jamaica has 14 parishes and ~3,400 km². Nigeria has 37 states, 774 LGAs, and ~923,000 km². The scientific question — can the model generalize from coarse supervision to fine spatial patterns? — is only meaningful at Nigeria's scale. Jamaica is too small to have enough regions for meaningful cross-validation.

2. **Data richness.** Nigeria has MICS6 2021 microdata (67,000+ household members), DHS 2018 (40,000 women, 31,000 children), LSMS 2018–2019 household surveys, NBS NLSS 2019 state poverty estimates, and ACLED conflict data. Jamaica's data ecosystem is much thinner.

3. **UNICEF priority.** Nigeria has the highest child deprivation burden in Africa. The operational value of sub-state predictions is far higher for Nigeria than for Jamaica.

The pipeline is fully config-driven. Jamaica (or any country with MICS + RWI data) can be re-activated by creating a `config/config_jam.yaml` and running `python main.py --country jam`.

---

## 3. Data Sources

### 3.1 Base Grid

| Item | Detail |
|---|---|
| **Source** | Meta / World Bank Relative Wealth Index dataset |
| **Resolution** | ~2.4 km cell centres (consistent with RWI sampling) |
| **Coverage** | All of Nigeria — 103,424 grid cells |
| **How joined** | Every other dataset is re-projected and sampled onto these cell centres |

### 3.2 Training Targets (Ground Truth)

| Dataset | Source | What it provides | Level |
|---|---|---|---|
| **MICS6 2021** | UNICEF Nigeria (SPSS microdata) | Multidimensional child deprivation — moderate & severe prevalence, deprivation depth | 37 states |
| **MICS6 Zone Targets** | Computed from MICS6 | Same indicators aggregated to 6 geopolitical zones | 6 zones |
| **MICS6 Urban/Rural** | Computed from MICS6 | State × urban/rural breakdown | 74 strata |

The MICS deprivation measure counts a child as moderately deprived if they lack access in ≥2 of 6 dimensions: nutrition, health, water, sanitation, shelter, education.

### 3.3 Geospatial Proxy Features

| Dataset | Source | Resolution | What it provides |
|---|---|---|---|
| **RWI** | Meta AI / World Bank | ~2.4 km | Composite relative wealth score per cell |
| **WorldPop 2020** | University of Southampton | 100 m → resampled | Children under-5 population per cell |
| **GHSL SMOD** | EU Joint Research Centre | 1 km | Settlement type (uninhabited → urban centre) |
| **GHSL Built Surface 2020** | EU JRC | 30 arc-sec | Satellite-measured fraction of cell covered by buildings |
| **VIIRS Nightlights 2019** | NASA | ~500 m | Night light intensity — proxy for economic activity |
| **Weiss Accessibility 2019** | Oxford / Google | 1 km | Travel time to nearest city (two thresholds) |
| **TerraClimate Rainfall 2018** | Climatology Lab | ~4 km | Annual precipitation in mm |
| **ACLED Conflict Events** | Armed Conflict Location & Event Data | Point → gridded | Conflict events and fatalities per area |
| **OSM Schools** | OpenStreetMap | Point → distance | Distance to nearest school (km) |
| **GRID3 + OSM Health Facilities** | HDX / OpenStreetMap | Point → distance | Distance to nearest health facility (km) |
| **OSM Building Density** | OpenStreetMap | Point → gridded | Building count per km² |
| **GADM Boundaries v4.1** | GADM | Polygon | State (ADM1) and LGA (ADM2) admin polygons |

### 3.4 Survey Microdata Features (new)

These are **not** raster or point data — they come from processing survey microdata at state × urban/rural level and joining them to every grid cell in that stratum.

| Dataset | Source | Variables extracted | Level |
|---|---|---|---|
| **MICS6 Education (hl.sav)** | MICS6 household listing file — 67,722 school-age members | School attendance rate, ever-attended rate, public school rate | State × urban/rural |
| **MICS6 Health (wm.sav)** | MICS6 women's file — 40,326 women | ANC rate, skilled delivery rate, facility delivery rate | State × urban/rural |
| **MICS6 Child Health (ch.sav)** | MICS6 children's file — 31,103 children | Vaccination card rate, diarrhea care-seeking rate | State × urban/rural |

### 3.5 Independent Validation Sources (not used in training)

| Dataset | Source | What it provides |
|---|---|---|
| **LSMS-ISA 2018–2019** | World Bank | 4,976 household consumption measurements with GPS — used for external spatial validation |
| **Nigeria DHS 2018 (flat files)** | DHS Program | 30,713 children under-5 across 1,389 clusters — zone-level deprivation computed; GPS pending for full integration |
| **NBS NLSS 2019** | National Bureau of Statistics Nigeria | State-level monetary poverty headcount for 36 states — third independent validation source |

---

## 4. Consolidated Dataset — Column Reference

There are three output tables. Here is every column and where it comes from.

### 4.1 Modeling Table
**File:** `Data/interim/nga/nga_modeling_table.parquet`
**Shape:** 103,424 rows × 46 columns — one row per ~1 km² grid cell

#### Identity & Location
| Column | Type | Source | Description |
|---|---|---|---|
| `cell_id` | int | Pipeline | Unique integer ID for each grid cell |
| `latitude` | float | RWI grid | Cell centre latitude (WGS84) |
| `longitude` | float | RWI grid | Cell centre longitude (WGS84) |

#### Administrative
| Column | Type | Source | Description |
|---|---|---|---|
| `subregion` | str | GADM spatial join | State name (37 states) |
| `gid_1` | str | GADM | GADM ADM1 identifier (e.g. NGA.4_1) |
| `geopolitical_zone` | str | Admin mappings | One of 6 zones (North Central, North East, North West, South East, South South, South West) |
| `state_urban_rural` | str | Derived | Concatenated state + urban/rural label (e.g. Kano_Urban) |
| `is_urban` | int | GHSL SMOD | 1 = urban, 0 = rural |
| `smod_class` | int | GHSL SMOD | Settlement class (10=uninhabited, 11=very low density rural, 21=suburban, 30=urban centre) |
| `smod_label` | str | GHSL SMOD | Human-readable settlement label |
| `parish_name` | str | GADM | State name (redundant with subregion, kept for compatibility) |
| `parish_imputed` | bool | Pipeline | True if state was spatially imputed (cell fell outside all state polygons) |

#### Target Variables (MICS Ground Truth)
| Column | Type | Source | Description |
|---|---|---|---|
| `moderate_prevalence` | float | MICS6 2021 | % of children deprived in ≥2 of 6 dimensions — the primary training target |
| `severe_prevalence` | float | MICS6 2021 | % of children deprived in ≥3 of 6 dimensions |
| `moderate_depth` | float | MICS6 2021 | Average number of deprivations among deprived children (intensity) |
| `severe_depth` | float | MICS6 2021 | Deprivation depth for severely deprived children |
| `in_modeling_sample` | bool | Pipeline | True if cell has a valid target and is used in training |

#### Wealth & Economic Features
| Column | Type | Source | Range | Description |
|---|---|---|---|---|
| `rwi` | float | Meta / World Bank RWI | −1.6 to +2.0 | Relative Wealth Index — higher = wealthier |
| `rwi_error` | float | Meta / World Bank RWI | 0.1 to 0.8 | Posterior uncertainty of RWI estimate |
| `nightlights` | float | VIIRS 2019 | 0 to 1,352 | Night light radiance — proxy for economic activity |
| `log_nightlights` | float | Derived | 0 to 7.2 | log1p(nightlights) — stabilizes right-skewed distribution |

#### Population Features
| Column | Type | Source | Range | Description |
|---|---|---|---|---|
| `population` | float | WorldPop 2020 | 0 to 1,206 | Estimated children under-5 per cell |
| `population_imputed` | int | Pipeline | 0 or 1 | 1 if population was spatially imputed |
| `log_population` | float | Derived | 0 to 7.1 | log1p(population) |

#### Accessibility Features
| Column | Type | Source | Range | Description |
|---|---|---|---|---|
| `travel_time_cities` | float | Weiss et al. | 0 to 753 min | Minutes walking/driving to nearest city >50k |
| `travel_time_50k` | float | Weiss et al. | 0 to 2,089 min | Alternative travel time threshold |
| `log_travel_time_cities` | float | Derived | 0 to 6.6 | log1p(travel_time_cities) |
| `log_travel_time_50k` | float | Derived | 0 to 7.6 | log1p(travel_time_50k) |
| `dist_school_km` | float | OSM Schools | 0.006 to 217 km | Distance to nearest school |
| `dist_health_km` | float | GRID3 + OSM | 0.006 to 31 km | Distance to nearest health facility |

#### Built Environment Features
| Column | Type | Source | Range | Description |
|---|---|---|---|---|
| `building_density` | float | OSM Buildings | 0 to 22,423 | Building count per km² |
| `log_building_density` | float | Derived | 0 to 10.0 | log1p(building_density) |
| `ghsl_built_frac` | float | GHSL 2020 | 0 to 0.70 | Fraction of cell covered by built-up surface |
| `ghsl_built_m2` | float | GHSL 2020 | 0 to large | Absolute built surface area in m² |
| `log_ghsl_built` | float | Derived | 0 to 13.2 | log1p(ghsl_built_m2) |

#### Environment & Conflict Features
| Column | Type | Source | Range | Description |
|---|---|---|---|---|
| `rainfall_mm` | float | TerraClimate 2018 | 0 to 4,306 mm | Annual precipitation — proxy for agricultural livelihood |
| `conflict_events` | float | ACLED (log-scaled) | 0 to 8.8 | Armed conflict events in surrounding area |
| `conflict_fatalities` | float | ACLED (log-scaled) | 0 to 10.8 | Conflict fatalities in surrounding area |

#### Education Features (from MICS6 microdata)
*State × urban/rural level — all cells in the same state-urban stratum share the same value*

| Column | Type | Source | Range | Description |
|---|---|---|---|---|
| `school_attendance_rate` | float | MICS6 hl.sav | 79% to 98% | Fraction of children aged 6–17 attending school this year |
| `ever_attended_rate` | float | MICS6 hl.sav | 31% to 100% | Fraction who ever attended any school |
| `public_school_rate` | float | MICS6 hl.sav | 84% to 100% | Fraction attending public (state-funded) school |

#### Health Utilization Features (from MICS6 microdata)
*State × urban/rural level — all cells in the same state-urban stratum share the same value*

| Column | Type | Source | Range | Description |
|---|---|---|---|---|
| `anc_rate` | float | MICS6 wm.sav | 27% to 100% | Fraction of women who received any antenatal care |
| `skilled_delivery_rate` | float | MICS6 wm.sav | 3% to 22% | Fraction delivered by doctor or nurse/midwife |
| `facility_delivery_rate` | float | MICS6 wm.sav | 3% to 21% | Fraction who gave birth in a health facility |
| `vacc_card_rate` | float | MICS6 ch.sav | 45% to 100% | Fraction of children with a vaccination card |
| `diarrhea_care_rate` | float | MICS6 ch.sav | 11% to 100% | Fraction who sought care when child had diarrhea |

---

### 4.2 Predictions Table
**File:** `Data/outputs/nga/tables/nga_predictions.parquet`
**Shape:** 103,424 rows × 36 columns

Same grid cells with model outputs added. Key additional columns:

| Column | Description |
|---|---|
| `ridge_moderate` | Ridge regression — moderate deprivation prediction, after reconciliation (%) |
| `ridge_severe` | Ridge — severe deprivation prediction (%) |
| `ridge_moderate_depth` | Ridge — deprivation depth prediction |
| `ridge_moderate_lower` | Ridge — 90% CI lower bound |
| `ridge_moderate_upper` | Ridge — 90% CI upper bound |
| `gam_moderate` | GAM — moderate deprivation prediction (%) |
| `gam_severe` | GAM — severe deprivation prediction (%) |
| `rwi_moderate` | RWI-only baseline prediction |
| `heuristic_moderate` | Urban/rural wealth tercile heuristic baseline |
| `uniform_moderate` | Uniform baseline (all cells in a state get the state mean) |

---

### 4.3 LGA Table
**File:** `Data/outputs/nga/tables/nga_lga_predictions.csv`
**Shape:** 775 rows × 17 columns — one row per Local Government Area

| Column | Description |
|---|---|
| `lga_id` | GADM ADM2 identifier |
| `state` | State name |
| `lga_name` | LGA name |
| `n_cells` | Number of grid cells in this LGA |
| `total_population` | Total child population (sum of cell populations) |
| `pct_urban` | Percentage of cells classified urban |
| `mics_state_truth` | MICS ground truth for the parent state (same for all LGAs in a state) |
| `ridge_moderate` | Population-weighted Ridge prediction for this LGA |
| `gam_moderate` | Population-weighted GAM prediction |
| `ridge_moderate_lower/upper` | Uncertainty bounds |

---

### 4.4 DHS 2018 Cluster Geo (GPS joined)
**Files:** `Data/Nigeria/dhs/nga_dhs_cluster_deprivation_geo.csv` and `.geojson`  
**Shape:** ~1,382 rows (valid coordinates) — one row per DHS cluster with survey-based deprivation + official GPS

| Column | Description |
|---|---|
| `cluster_id` | Matches `v001` in DHS recode and `DHSCLUST` in `NGGE7BFL.shp` |
| `LATNUM`, `LONGNUM` | Displaced coordinates (WGS84); urban up to ~2 km, rural up to ~5 km jitter per DHS policy |
| `ADM1NAME` | State name from DHS admin fields |
| `URBAN_RURA` | `U` / `R` — DHS urban/rural stratum |
| `deprivation_index`, `stunting_rate`, … | Same measures as in `process_dhs.py` cluster output |

**Scripts:** `python -m src.scripts.merge_dhs_gps` (after `process_dhs` and with `Data/Nigeria/dhs/raw/NGGE7BFL/NGGE7BFL.shp` available).

---

## 5. How We Train the Model

### Step 1 — Build the Grid
The RWI dataset provides 103,424 point locations covering Nigeria. These become the spatial unit of analysis — every other dataset is snapped to these points.

### Step 2 — Sample All Features
For each of the 103,424 cells, we extract values from every data source:
- Rasters (nightlights, rainfall, GHSL): bilinear interpolation at each cell centre
- Point datasets (schools, health facilities, buildings): spatial indexing to compute distance or density within a radius
- Polygon datasets (GADM state boundaries): spatial join to assign each cell to a state
- Survey microdata (MICS education/health): state × urban/rural level join

### Step 3 — Assign Targets
Each cell inherits the MICS deprivation prevalence of its state. Every cell in Kano gets `moderate_prevalence = 50.05`. This is the **coarse supervision signal** — the model does not see individual cell-level deprivation; it only knows the state total.

### Step 4 — Train the Model
The model sees each cell's 28-feature vector and outputs a raw deprivation score. Training works through **aggregation loss**:

```
For each state r:
  1. Run model on every cell i inside r → get predicted score ŷᵢ
  2. Population-weight-average all predictions:
     Ŷᵣ = Σ(popᵢ × ŷᵢ) / Σ(popᵢ)
  3. Compare to MICS truth: loss = (Ŷᵣ − Yᵣ)²
  4. Backpropagate — adjust weights to minimize this loss
```

The model **never sees** "this cell should be X%." It only learns "all cells in this state together should average to X%." This is weak supervision — the signal is coarse but real.

### Models Trained

| Model | Method | Key property |
|---|---|---|
| **Ridge Regression** | L2-regularised linear model | Fast, interpretable, cross-validated regularisation strength |
| **GAM** | Generalised Additive Model with splines | Captures non-linear feature effects, one smooth per feature |
| **WSNN** | Weakly Supervised Neural Network (PyTorch, 3-layer MLP) | Most flexible, learns feature interactions |
| **GBM** | LightGBM gradient boosted trees | Best feature importance; run separately due to threading constraints |
| **Uniform** | Baseline — state average assigned to every cell | No model, just the coarse truth spread evenly |
| **Heuristic** | Baseline — wealth tercile multipliers within state | Simple rule-based redistribution |
| **RWI** | Baseline — inverted RWI used directly as deprivation proxy | No training needed |

---

## 6. How We Make Predictions

### Step 1 — Raw Prediction
The trained model scores every cell. At this point the aggregate across a state may not exactly match the MICS truth — the model has generalised from training states and may be slightly off on any given state.

```
Example before reconciliation (Ridge, LOZO test):
  Benue truth = 62.2%
  Ridge predicts aggregate = 42.3%   ← 19.9 pp off
```

### Step 2 — Hard Administrative Reconciliation
A post-processing scaling step forces exact consistency with the MICS state targets:

```
For each state:
  scale_factor = MICS_truth / population_weighted_mean(raw_predictions)
  reconciled_prediction_i = raw_prediction_i × scale_factor
```

After this step, the population-weighted average of every cell within a state **exactly** equals the MICS truth — to within 0.01 percentage points. This is verified and logged for all 37 states on every run.

```
After reconciliation:
  Benue aggregate = 62.2%  ✓ (diff = 0.000000)
```

### Step 3 — Aggregate to LGAs
Grid cell predictions are spatially joined to 775 LGA polygons and population-weighted to produce one score per LGA. These are the most actionable outputs for program targeting.

### Step 4 — Uncertainty Quantification
For Ridge, a 90% confidence interval is computed by propagating the RWI posterior error through the model: 100 Monte Carlo draws of each cell's RWI → 100 sets of predictions → 5th/95th percentile bounds.

---

## 7. How We Validate Accuracy

### Leave-One-Zone-Out (LOZO) Cross-Validation
The gold-standard test: train on 35 states, predict the held-out state, measure error **before reconciliation**. This answers: "how accurately can the model predict a region it has never seen?"

Repeated for all 36 states (Borno excluded — no LSMS coverage).

### Hierarchical Cross-Level Validation
Train on 6 geopolitical zones (coarser level), evaluate whether 37 state patterns emerge without any state-level supervision. This tests true spatial generalisation.

### Three Independent Validation Sources

| Source | Type | What it validates |
|---|---|---|
| LSMS-ISA 2018–2019 | 4,976 household consumption GPS points | Whether cell-level predictions correlate with actual household welfare |
| DHS 2018 | Zone-level + **~1,382 geolocated clusters** (shapefile `NGGE7BFL`) | Zone tables vs MICS; **cluster-level:** Ridge/RWI at nearest grid cell vs DHS composite index (`validate_predictions_vs_dhs_gps.py`) |
| NBS NLSS 2019 | State monetary poverty (36 states) | Whether model state rankings agree with consumption-based poverty |

---

## 8. Current Results

### LOZO Accuracy (before reconciliation — the honest test)

| Model | Mean Error | Median Error | Within 5 pp | Within 10 pp |
|---|---|---|---|---|
| **WSNN** | **8.3 pp** | **5.8 pp** | **44%** | **64%** |
| Ridge | 11.6 pp | 7.7 pp | 28% | 56% |
| GAM | 158 pp (outlier-driven) | 13.2 pp | 25% | 36% |
| RWI baseline | 15.5 pp | 17.0 pp | 11% | 33% |

The WSNN is the best-performing model. The ML models roughly halve the error of the RWI baseline.

### Cross-Survey Agreement

| Comparison | Pearson r | Spearman ρ | Interpretation |
|---|---|---|---|
| DHS 2018 vs MICS 2021 (zone level) | 0.96 | 0.77 | Strong — both surveys rank zones consistently |
| NBS NLSS 2019 vs MICS 2021 (state level) | 0.67 | 0.64 | Moderate — different poverty concepts explain divergence |

### Reconciliation
All 37 states reconcile to MICS truth with `diff = 0.000000` — the hierarchical guarantee is exact.

### DHS GPS — cluster vs grid (after `NGGE7BFL` + merge)
Nearest grid cell to each DHS point (~1 km typical). DHS `deprivation_index` scaled ×100 for comparability to model % (different constructs; interpret as association, not equality).

| Metric | Ridge (reconciled) | RWI |
|---|---|---|
| Spearman ρ vs DHS index×100 | ~0.41 | ~0.54 |
| Pearson r | ~0.38 | — |
| MAE (index×100 vs prediction %) | ~16 pp | — |

RWI can track the DHS composite more closely at cluster level because it is a smooth wealth map; Ridge is pulled toward MICS state targets and may underfit local DHS–only variation.

---

## 9. What Is Left

### DHS GPS — done for validation; optional training upgrade
1. `merge_dhs_gps` + `validate_predictions_vs_dhs_gps` are implemented.  
2. **Optional next step:** add cluster-level (or soft) loss using DHS labels alongside MICS state loss — retrain with `main.py --force-rerun` after extending the training objective (not yet in `main.py`).

### LOZO still the main pre-reconciliation test
Expected further MAE drop if DHS cluster targets are used **inside** the loss (point-level or pseudo-labels) — that is a separate code change from validation-only GPS use.

### Ready to Implement (No Blockers)
| Item | Effort | Impact |
|---|---|---|
| LGA-level governance / EMIS school quality features | 2–3 days | Medium |
| Second country pipeline (Ethiopia, Sudan, Mali) | 1–2 days | High strategic |

### Pipeline Commands

```bash
# Full run (Ridge + GAM, skip slow GBM)
python main.py --country nga --skip-gbm --phase all

# Force rebuild from scratch
python main.py --country nga --force-rerun --skip-gbm

# Run only evaluation
python main.py --country nga --phase eval

# Run only LOZO cross-validation
python main.py --country nga --phase lozo

# DHS: cluster deprivation from flat files + GPS merge + vs-grid validation
python -m src.scripts.process_dhs
python -m src.scripts.merge_dhs_gps
python -m src.scripts.validate_predictions_vs_dhs_gps
```
