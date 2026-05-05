# How to run the application — outputs and what they mean

> **Research tool only.** This pipeline is **not** an official statistics system. Outputs are spatial disaggregations consistent with survey totals at the chosen administrative level, intended for research and prioritisation — not for publishing headline national poverty rates without independent review.

For architecture and status, see **`README.md`**, **`TECHNICAL_OVERVIEW.md`**, and **`PROJECT_STATUS.md`**. This file also covers **demonstration maps (§5.1)**, **reconciliation vs raw accuracy (§7)**, and **which eval CSVs mean what**.

**Full walkthrough of every Nigeria map + table (what they show, how to read them, what they’re for):** **`OUTPUTS_GUIDE_NGA.md`**.

---

## 1. Prerequisites

**Python environment**

```bash
cd /path/to/BorealisAI_UNICEF
pip install -r requirements.txt
```

Optional model stacks (skip with flags if not installed):

- **GAM:** `pygam`
- **WSNN:** `torch`

**Nigeria config**

The Nigeria run uses **`config/config_nga.yaml`**. Paths inside it are relative to the project root.

**Data you need**

- **Minimum for a fast run using caches:** the repo’s cached Parquet/CSV intermediates under `Data/interim/nga/` and feature files under `Data/Nigeria/features/` (as already documented in `README.md`).
- **To rebuild from scratch:** MICS6 `.sav` files, population raster, GADM, RWI grid, optional DHS flat files under `Data/Nigeria/dhs/raw/`, etc. Missing optional inputs usually skip quietly when caches exist; `--force-rerun` will fail if a required raw file is absent.

---

## 2. Main entry point: `main.py`

### 2.1 Nigeria — full pipeline (typical)

```bash
python main.py --country nga
```

Runs: **data pipeline (steps 01–05)** → **baselines** → **models** (Ridge + optional GBM/GAM/WSNN) → **evaluation** → **save tables, maps, eval CSVs**.

### 2.2 Faster iteration

```bash
# Skip heavier models (good for debugging)
python main.py --country nga --skip-gbm --skip-gam --skip-wsnn

# Ridge-only composite run (example)
python main.py --country nga --skip-gbm --skip-gam --skip-wsnn
```

### 2.3 Phases (stop after a stage)

| Phase | Command | What runs |
|-------|---------|-----------|
| Data only | `python main.py --country nga --phase data` | Builds/refreshes interim grid, proxies, admin, targets, **modeling table** |
| Through baselines | `--phase baselines` | Data + uniform / heuristic / RWI baselines |
| Through models | `--phase models` | Above + trained models |
| Full | `--phase all` *(default)* | Above + evaluation + disk outputs |

### 2.4 Force recomputation

```bash
python main.py --country nga --force-rerun
```

Recomputes cached steps even if interim files exist. Use when you change raw inputs, config features, or code.

### 2.5 Custom config path

```bash
python main.py --country nga --config /absolute/path/to/config.yaml
```

*(Default without `--country` uses Jamaica / `config.yaml` — see repo defaults.)*

---

## 3. Where outputs go (Nigeria)

All paths below are under the project root unless noted.

| Directory | Role |
|-----------|------|
| **`Data/interim/nga/`** | Cached pipeline artifacts (grid, proxies, admin assignment, MICS targets, modeling Parquet). Safe to delete specific files to force a step to rerun. |
| **`Data/outputs/nga/tables/`** | Final **tabular** predictions and breakdowns (cells and sometimes LGA rollup). |
| **`Data/outputs/nga/maps/`** | **Folium HTML** maps (and optional PNG). Open in a browser. |
| **`Data/outputs/nga/eval/`** | **Metrics, CV, validation** CSVs and text summaries. |

---

## 4. Tables (`Data/outputs/nga/tables/`)

### 4.1 Core prediction tables

| File | Description |
|------|-------------|
| **`nga_predictions.parquet`** | Primary machine-readable export: one row per grid cell with predicted **composite** deprivation (moderate / severe) and all selected model columns. Prefer Parquet for size and types. |
| **`nga_predictions.csv`** | Same logical content as Parquet (may be large or gitignored if over repo limits). |
| **`nga_prediction_breakdown.csv`** | Per-cell **Ridge linear decomposition** (β·z-style contributions), themes, and raw feature snapshots where enabled — used for explainability and map popups. |
| **`nga_full_consolidated.parquet`** | Wide consolidated table merging modeling features with predictions where the pipeline is configured to produce it — useful for analysis notebooks. |
| **`nga_lga_predictions.csv`** | **LGA-level** (ADM2) rollups: population-weighted averages of cell predictions within each Local Government Area polygon. |
| **`nga_dimension_predictions.csv`** | Per-dimension Ridge predictions (shelter, water, nutrition, education, health, etc.) at **cell** level — produced by the dimension script (see **§9**), not by `main.py` alone. |

**Typical column meanings (composite model)**

- **`moderate_prevalence` / `severe_prevalence`** — Reconciled **composite** child deprivation prevalence targets at state level (from MICS), used as supervision; after reconciliation, cell predictions aggregate back to these totals **per admin zone** in the config.
- **`ridge_moderate` / `ridge_severe`** — Ridge model predictions (%), usually **reconciled** to match official zone totals.
- **`rwi_moderate` / `rwi_severe`** — Baseline that redistributes state targets using **RWI** only (no learned weights).
- **`uniform_moderate` / `uniform_severe`** — Baseline that spreads each zone’s rate **uniformly** across its cells.
- **`gbm_moderate` / `gam_moderate` / `wsnn_moderate`** — Optional models when not skipped; same scale as prevalence (%).
- **`*_lower` / `*_upper`** — Uncertainty intervals where the pipeline computes bootstrap or analytic CIs (model-dependent).
- **`ridge_bdg_*` / theme columns** — Packed explainability fields (dominant theme, signed contributions) when breakdown merge is enabled.

Exact column sets depend on config and which models ran successfully.

---

## 5. Maps (`Data/outputs/nga/maps/`)

Open **`.html`** files in Chrome, Firefox, or Safari (double-click or “Open with browser”).

| File | What it shows |
|------|----------------|
| **`nga_predictions_map_sample.html`** | **Recommended default for exploration** — subset of cells with clustering/opacity/legend so the map stays readable. |
| **`nga_predictions_map.html`** | Full-grid Folium export — **very large**; many overlapping circles; same semantics as sample but dense. |
| **`nga_comparison_map.html`** | Side-by-side or layered view comparing **truth vs prediction** at **LGA** or comparable resolution (built from consolidated predictions + boundaries). Good for stakeholder PDFs / screenshots. |
| **`nga_uncertainty_map.html`** — Where CIs exist — maps width or bounds of uncertainty for the chosen model. |
| **`nga_dimension_*_map.html`** | One map per **Kyriaki dimension** (e.g. shelter, nutrition, health). |
| **`nga_dimension_comparison_map.html`** | **Multi-panel** comparison of dimensions (+ composite where included). |
| **`nga_lga_deprivation_map.png`** | Static raster export when generated (quick slides). |

**How to read circle maps**

- Each point is a **grid cell centre** (aligned to the RWI Nigeria grid, ~2.4 km spacing — see `TECHNICAL_OVERVIEW.md`).
- **Colour / radius** encode predicted deprivation or uncertainty depending on layer settings (see map legend).
- At full density, overlap is normal — use **`nga_predictions_map_sample.html`** or LGA maps for communication.

### 5.1 HTML files to use in demonstrations (priority order)

All paths: `Data/outputs/nga/maps/`. Open in a normal browser. For **live demos or screen sharing**, prefer **LGA** and **sampled** maps — full-cell maps can freeze or clutter the screen.

| Priority | File | Use when |
|:--------:|------|----------|
| **1** | **`nga_comparison_map.html`** | **Default stakeholder map:** LGA polygons, truth vs Ridge (or baseline), readable at scale — best for presentations and PDFs. |
| **2** | **`nga_predictions_map_sample.html`** | **Composite** deprivation at grid cells — clustered/sampled points with legend; interactive without opening the multi‑million‑line full grid map. |
| **3** | **`nga_dimension_comparison_map.html`** | **Multi-panel:** Kyriaki dimensions (+ composite where included) side by side — shows education / shelter / nutrition **themes** in one view. |
| **4** | **`nga_uncertainty_map.html`** | When you need to discuss **confidence / CI width** for the chosen model. |
| **5** | **`nga_dimension_<name>_map.html`** | Deep dive on **one** dimension only (e.g. `nga_dimension_nutrition_map.html`, `nga_dimension_edu_5_14_map.html`). Regenerate with `run_dimension_models.py` if a file looks empty or stale. |
| **6** | **`nga_predictions_map.html`** | Full **103k-cell** Folium export — **very large**; use only if you need every cell and accept slow load; often excluded from Git (`GH001`). |

Optional / rebuild: **`nga_explainability_map.html`** if your pipeline run produced it — explainability-first overlay (see `PROJECT_STATUS.md`). **`nga_lga_predictions.geojson`** pairs with GIS tools; **`nga_lga_deprivation_map.png`** is a static slide asset when present.

---

## 6. Evaluation (`Data/outputs/nga/eval/`)

| File | Meaning |
|------|---------|
| **`evaluation_summary.csv`** | One row per **method** (uniform, heuristic, RWI, Ridge, GBM, GAM, …): **MAE**, **Pearson/Spearman** vs targets, spatial smoothness proxies, optional per-state CV columns, CI width summaries. |
| **`lozo_evaluation.csv`** | **Leave-one-state-out** (column label `zone` = **`subregion`** state): train on all other states, aggregate **raw** predictions on the held-out state vs target — **no reconciliation** on that hold-out — tests geographic generalisation. |
| **`hierarchical_validation.csv`** | **Cross-level** checks (e.g. train on coarse zones, predict states or finer strata). Contains **`reconciled=False`** (raw) and **`reconciled=True`** rows — compare MAE / Pearson for **without vs with** reconciliation. |
| **`hierarchical_validation_detail.csv`** | Row-level detail for hierarchical runs. |
| **`two_level_cv.csv`** | Two-level cross-validation metrics where configured. |
| **`significance_tests.csv`** | Paired / comparative tests between methods where implemented. |
| **`dhs_gps_validation.csv`** / **`dhs_gps_validation.txt`** | Compares predictions to **DHS cluster**-based indices at GPS locations (external consistency check; not MICS truth). |
| **`dhs_mics_crossvalidation.csv`** / **`.txt`** | Cross-survey comparison between DHS and MICS constructs. |
| **`nbs_model_comparison.csv`** / **`.txt`** | Comparison against **NBS** monetary poverty or related aggregates where wired in. |
| **`ridge_feature_contribution_breakdown.csv`** | Global Ridge **coefficients / grouped themes** (not geographic). |
| **`rwi_uncertainty_analysis.csv`** | Uncertainty diagnostics tied to RWI or redistributed baselines. |
| **`nga_dimension_summary.csv`** | Per-dimension Ridge summary from `run_dimension_models.py`; **`pred_mean`** is after **reconciliation** (see §7). |
| **`dhs_aux_stack_sweep.csv`**, **`dhs_soft_label_sweep.csv`** | Hyperparameter / ablation sweeps for DHS-assisted Ridge — optional artifacts from running sweep scripts. |
| **`admin_detail_*.csv`** | Per-admin breakdown of errors (**ridge**, **gbm**, **rwi**, etc.) and **depth** variants if depth targets are enabled. |
| **`gbm_feature_importances.csv`** | GBM global importances when GBM ran. |

---

## 7. Reconciliation vs raw predictions — why, and where “accuracy” lives

### Problem you are solving

- **Source of truth** exists only over a **large area** *A* (here: usually each **state’s** survey prevalence).
- You predict for **small regions** *B* (grid cells) that **tile** *A*.
- **Dimension** models add **thematic** scores (education, shelter, …) — separate targets and reconciliations from the **composite** multidimensional score (`main.py`).

There are **no** cell-level survey labels, so the ML mapping from satellite/survey-derived features to prevalence is **underdetermined**. **Reconciliation** rescales cell predictions **within each state** so that **population-weighted** averages match the official target for that score (composite or each dimension). That guarantees consistency with the only trusted totals.

### Why reconcile?

Without reconciliation, the map could look smooth but **every state’s aggregate** could be wrong vs MICS. Reconciliation answers: *“Given coarse truth for A, what allocation across B is consistent with features **and** sums back correctly?”*

### “Accuracy” without reconciliation (generalisation of the **pattern**)

These metrics reflect the model **before** forcing state totals (or compare raw vs reconciled explicitly):

| Question | Where to look |
|----------|----------------|
| Held-out **state**: how wrong is the **raw** Ridge aggregate? | **`lozo_evaluation.csv`** — Ridge row uses **unreconciled** cell predictions averaged into the held-out state. |
| Same for hierarchical experiments | **`hierarchical_validation.csv`** — rows with **`reconciled=False`** vs **`True`**. |
| Independent survey at cluster GPS | **`dhs_gps_validation.txt`** / **`.csv`** — not MICS reconciliation; checks spatial co-movement with DHS. |
| State-level Pearson *r* in **`evaluation_summary.csv`** | Mostly reflects **reconciled** agreement (composite MAE at admin level can be ~0 by construction); prefer LOZO / hierarchical raw / DHS for **unbiased** coarse accuracy. |

### Composite vs dimension breakdown

- **Composite** (`ridge_moderate`, …): trained on **MICS six-dimension** multidimensional deprivation (≥2 of six domains — see `compute_mics_deprivation.py`), reconciled to **composite** state targets.
- **Kyriaki dimensions** (`run_dimension_models.py`): **different** definitions (e.g. DHS HAZ for nutrition, split education/health). Each is reconciled to **its own** dimension state targets. They **do not** algebraically sum to the composite prevalence — different definitions and joint vs marginal structure.

### Dimension models: raw vs reconciled on disk

`nga_dimension_predictions.csv` stores **reconciled** `{dim}_moderate` columns. Raw Ridge scores exist only inside the script before `_reconcile`; to audit raw dimension error at state level, add a small export or inspect logs (`train_r` is in-sample fit, not LOZO).

---

## 8. Interim files worth knowing (`Data/interim/nga/`)

| File | Purpose |
|------|---------|
| **`nga_base_grid.parquet`** | Cell IDs, coordinates, RWI anchor. |
| **`nga_grid_proxies.parquet`** | Sampled rasters and distances (travel time, lights, …). |
| **`nga_grid_admin.parquet`** | State/LGA assignment per cell. |
| **`nga_targets*.csv`** | Aggregated **MICS** targets at state / zone / urban-rural levels. |
| **`nga_dimension_targets.csv`** | Per-dimension targets for Kyriaki dimensions (+ DHS HAZ merge for nutrition when available). |
| **`nga_modeling_table.parquet`** | Master learning table: features + targets + merges for modeling. |
| **`nga_dhs_haz_targets.csv`** | Cached **DHS HAZ** state nutrition targets when ingestion succeeds. |

Deleting a file here forces the next `main.py` run (or dimension script) to rebuild that stage **if** the code path needs it.

---

## 9. Optional scripts (not `main.py`)

Run from project root with `python`. Examples:

| Script | Purpose |
|--------|---------|
| **`python src/scripts/run_dimension_models.py --country nga`** | Recompute **per-dimension** targets/models/maps (`nga_dimension_predictions.csv`, dimension HTML maps, `nga_dimension_summary.csv`). Use `--no-maps` for speed. |
| **`python src/scripts/process_dhs.py`** | Rebuild DHS cluster deprivation CSVs from flat files (when raw data present). |
| **`python -m src.scripts.merge_dhs_gps`** | Join cluster table to GPS shapefile. |
| **`python -m src.scripts.validate_predictions_vs_dhs_gps`** | Refresh DHS GPS validation tables after predictions change. |
| **`python src/scripts/ingest_d1_features.py`** | Refresh NBS/NEMIS/IIAG merges into the modeling table path used by D1. |

See **`PROJECT_STATUS.md`** for the full backlog and **`config/config_nga.yaml`** for paths.

---

## 10. Quick troubleshooting

| Symptom | What to try |
|---------|-------------|
| Import / module errors | `pip install -r requirements.txt`; run from **repo root** so `src` imports resolve. |
| Missing raster / `.sav` | Run **without** `--force-rerun` to use caches; or add data under paths in `config_nga.yaml`. |
| Map too heavy to open | Use **`*_sample.html`** or LGA comparison maps. |
| Wrong country config | Pass **`--country nga`** explicitly. |

---

*Last updated: May 2026 — aligned with `config/config_nga.yaml` layout.*
