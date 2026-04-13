# UNICEF × RBC Borealis AI Project
## Data Inventory and Technical Notes

For the problem statement, ML formulation, and evaluation framework, see
`docs/problem_statement.pdf`.

For the complete technical implementation record — data inspection results,
pipeline steps, decisions made, and bugs fixed — see `pipeline_phases.md`.

---

## Data Inventory

### A. Outcome / Target Data
Administrative or subgroup-level child poverty / deprivation tables.

### B. Spatial Proxy Data
Grid-level geospatial features used to infer fine-scale variation.

### C. Linking / Boundary Data
Spatial boundaries needed to connect grid cells to official administrative regions.

---

### `UNICEF FDN - RBC LSI Spring 26.docx`
A technical concept / proposal document describing the scientific and methodological
framing of the project. Defines the research question, scientific framing, and
evaluation logic.

---

### `UNICEF FDN - RBC Borealis LSI Spring 2026 - Kick off.pptx`
A kickoff deck giving the operational context and problem motivation. Grounds the
project in a real humanitarian workflow and clarifies the use case for spatial
prioritization.

---

### `Child Poverty latest estimates Feb 2026 with disaggregations.xlsx`
A large cross-country administrative / survey-based child poverty table.

- Broad international coverage
- Multiple survey years (DHS and MICS sources)
- Subgroup disaggregations: sex, residence, wealth quintile, and combinations
- Measures: severe prevalence, moderate prevalence, depth, deprivation count distributions

This is the broad target-side reference table supporting outcome definition,
subgroup logic, country filtering, and cross-country validation design.

---

### `ChPov_JAM_CUB.xlsx`
A child poverty file focused on Jamaica and Cuba with subregional information.
Contains poverty-related outputs at finer subnational resolution (Urban/Rural/KMA).

Used as the primary poverty target for the Jamaica pilot.

---

### `jam_relative_wealth_index.csv`
A Jamaica-specific Relative Wealth Index file providing grid-level wealth proxy
values linked to coordinates.

Plays two roles:
1. Core model feature
2. Main baseline comparator (RWI redistribution baseline)

Chosen as the **base grid** onto which all other geospatial layers are aligned.
1,745 rows × 4 columns (latitude, longitude, rwi, error); ~2.3 km spacing; zero
missing values.

---

### `jam_pop_2030_CN_100m_R2025A_v1.tif`
A high-resolution Jamaica population raster (WorldPop). 100 m resolution,
EPSG:4326. NoData = −99999.

Required for population-weighted aggregation and reconciliation.

---

### `GHS_SMOD_E2030_GLOBE_R2023A_54009_1000_V2_0.zip`
GHSL SMOD settlement classification dataset. Global GeoTIFF, 1 km resolution,
ESRI:54009 Mollweide (requires CRS reprojection before sampling).

Provides urban/rural settlement structure, a strong predictor of deprivation
patterns and service access.

---

### `gadm41_JAM.gpkg`
Jamaica administrative boundary file (GeoPackage, EPSG:4326). Layers: ADM_0
(country), ADM_1 (14 parishes). No ADM_2 available.

The spatial linking layer that assigns grid cells to administrative units for
reconciliation.

---

### `cit_017_accessibility_to_cities.zip`
Travel-time-to-cities accessibility dataset. Global GeoTIFF, EPSG:4326.
Sentinel NoData = −9999. 79 missing values on the RWI grid after sampling.

Captures remoteness and structural access to services — one of the best
candidates for helping the model beat an RWI-only baseline.

---

### `access_50k.zip`
Travel time to nearest 50k-population centre. Global GeoTIFF, EPSG:4326.
Sentinel NoData = −9999. 86 missing values on the RWI grid after sampling.

---

## Major Risks

**Risk 1 — You might just recreate RWI**
If the model adds no value beyond RWI, that is still a valid research result.

**Risk 2 — Region matching may be messy**
Administrative names / codes across datasets may not line up cleanly.

**Risk 3 — Feature-resolution mismatch**
Different rasters may not align naturally.

**Risk 4 — False precision**
A fine-resolution map may look more certain than it actually is.

**Risk 5 — Weak evaluation design**
If the validation setup is weak, claims about model improvement will not hold.
