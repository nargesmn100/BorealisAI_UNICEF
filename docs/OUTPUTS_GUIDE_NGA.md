# Nigeria outputs guide — maps and tables

> **Research tool only.** Not official statistics.  
> Paths are relative to the project root.

This document explains **what each major Nigeria artifact shows**, **how to read it**, and **why it is useful**. For run commands see **`HOW_TO_RUN.md`**.

---

## 1. Interactive maps (`Data/outputs/nga/maps/`)

Open `.html` files in a normal browser. Prefer **sample** and **LGA** maps for demos; **full-grid** HTML embeds hundreds of thousands of points and can be slow.

### 1.1 Composite deprivation (general “poverty score” map)

These show **multidimensional child deprivation** — predicted **moderate** prevalence (%), same family as MICS composite supervision (not monetary poverty).

| File | What it shows | How to read it | Why it’s useful |
|------|----------------|----------------|-----------------|
| **`nga_predictions_map_sample.html`** | **Subset** of grid cells (~5k) with MarkerCluster, opacity, legend. Colours (or circle style) encode **Ridge** `ridge_moderate` (or first available model in priority order — see `main.py`). | Pan/zoom; **hover** for lat/lon; **click** for full popup. **Legend** = % moderate deprivation. | **Default for live demos** — loads fast, readable, same geography as full map. |

#### Reading the prediction popup (click a cell)

| Block | What it means |
|--------|----------------|
| **LGA / State** (`parish_name` / `subregion`) | Admin labels from the grid join — e.g. which LGA and state contain this grid point. |
| **Cell ID** | Integer row id in the modelling table (optional line). |
| **Lat / Lon** | WGS84 coordinates of the **grid point** (5 decimals ≈ 1 m). Use for GIS, Google Earth, or to cite an exact location. |
| **RWI** | Relative Wealth Index at this point (higher = wealthier in the Meta/WB sense). |
| **Pop** | Population allocated to this grid cell from WorldPop sampling — can be **0** in very sparsely modelled cells. |
| **Moderate poverty** | Modelled **moderate multidimensional child deprivation %** for that cell (after state-level reconciliation). |
| **Ridge (linear) explain** | Short **explainability** block: themes group features (e.g. DHS cluster, wealth, hazards); **Top features** list the largest linear contributions (β×standardised value–style). Positive = pushes deprivation **up**; negative = pushes **down**. Same idea as `nga_prediction_breakdown.csv`. |
| **`nga_predictions_map.html`** | **All** ~100k+ cell centres as circles — full composite map. | Same as sample but **every** cell; heavy overlap. Use legend and popups. | **Full detail** for screenshots of a subregion you zoom into, or when you need a specific cell; often huge file. |
| **`nga_comparison_map.html`** *(if present after a full run)* | **LGA polygons** with model vs “truth” / baselines in tooltips. | **Hover** LGAs for **state target vs Ridge (and sometimes RWI)**. Read as **areal** summary, not 100k dots. | **Best stakeholder slide** — clear admin shapes, easy to compare model to survey at LGA scale. |
| **`nga_uncertainty_map.html`** | **Uncertainty** (CI width or interval) for the **selected** model when CIs were computed. | **Wider** coloured bands or darker = **more uncertain** cell predictions (exact encoding follows legend on map). | Discussing **confidence**; prioritising verification field visits. |

### 1.2 Dimension maps (Kyriaki themes)

| File | What it shows | How to read it | Why it’s useful |
|------|----------------|----------------|-----------------|
| **`nga_dimension_comparison_map.html`** | **Nine Leaflet panels**: shelter, sanitation, water, nutrition, education bands, health bands, **Composite (Ridge)**. **LGA** polygons, % per panel. | **In-page “How to read this map”** block at top. Each panel: **own colour scale**; **%** = predicted **moderate** prevalence for **that dimension only** in that LGA. | **One file** to show **theme comparison** in a meeting. |
| **`nga_dimension_<name>_map.html`** | **Single** dimension at cell or sample level (e.g. `nga_dimension_water_map.html`). | Same colour logic as other Folium maps; check title. | **Deep dive** on one policy theme. **Regenerate** with `run_dimension_models.py` if a file is empty or stale. |
| **`nga_lga_deprivation_map.png`** | **Static** image (if generated). | As any PNG. | **Slides** without a browser. |
| **`nga_lga_predictions.geojson`** | **Vector** LGA layer + properties (not HTML). | Open in QGIS, kepler, etc. | **GIS** workflows, custom styling, printing. |

---

## 2. Tables — cell and LGA predictions (`Data/outputs/nga/tables/`)

These are **tabular** outputs from `main.py` (except **`nga_dimension_predictions.csv`**, which needs **`run_dimension_models.py`**).

| File | What it shows | How to read it | Why it’s useful |
|------|----------------|----------------|-----------------|
| **`nga_predictions.parquet`** | **One row per grid cell**: coordinates, population, targets (`moderate_prevalence`, …), **`ridge_moderate`**, other models, optional CIs. | Use **Python/pandas** or Parquet-aware tools. Columns are documented in **`TECHNICAL_OVERVIEW.md`** / config. Primary key is usually cell index or `cell_id` + lat/lon. | **Main analysis table** — smaller than CSV, typed columns. |
| **`nga_predictions.csv`** | Same logical content as Parquet (sometimes omitted from Git if >100 MB). | Spreadsheet or `pandas.read_csv` — **avoid** opening 100k-row CSV in Excel for memory. | Interop with tools that don’t read Parquet. |
| **`nga_prediction_breakdown.csv`** | **Explainability**: Ridge **linear** contributions (β·z-style), **themes**, raw feature snapshots where enabled. | Each row aligns with a cell; columns like `ridge_bdg_*`, `ridge_theme__*`. | **Why** the model scored a cell high/low; policy narrative + map popups. |
| **`nga_full_consolidated.parquet`** | **Wide** merge: **features + predictions** (and sometimes breakdown columns). | Filter by `subregion`, join keys as in pipeline. | **Research notebooks** — one file for regressions / correlations. |
| **`nga_lga_predictions.csv`** | **775 LGAs**: population-weighted **rollups** of cell predictions + often **MICS state truth** column for context. | Rows = LGA; columns include `ridge_moderate`, dimension `*_moderate` if merged, `total_population`, `state`. | **Sub-state reporting**; feeds **comparison map** and stakeholder tables. |
| **`nga_dimension_predictions.csv`** | **Per-dimension** moderate % at **cell** level (`shelter_moderate`, …). | Join on cell index with `nga_predictions` / grid. Values are **reconciled** per dimension (see **`HOW_TO_RUN.md` §7**). | **Thematic** maps and analysis when you need education vs water separately. |

---

## 3. Evaluation outputs (`Data/outputs/nga/eval/`)

CSV/text files from **evaluation phases**. Row counts and exact columns can vary slightly by run options (e.g. skipping WSNN).

### 3.1 Headline comparison across methods

| File | What it shows | How to read it | Why it’s useful |
|------|----------------|----------------|-----------------|
| **`evaluation_summary.csv`** | **One row per method** (uniform, heuristic, RWI, Ridge, GBM, …): **MAE**, **Pearson/Spearman** vs targets, smoothness, optional per-admin CV columns. | Higher **Pearson *r*** = better rank agreement with **state** targets **after** reconciliation for learned models. **Uniform/heuristic** interpret differently. | **Single table** to quote “accuracy” of each approach. |

### 3.2 Generalisation (held-out geography)

| File | What it shows | How to read it | Why it’s useful |
|------|----------------|----------------|-----------------|
| **`lozo_evaluation.csv`** | **Leave-one-state-out**: for each held-out **state**, **target** vs **population-weighted mean prediction** on that state’s cells — **Ridge uses raw** preds (no reconciliation on hold-out). | Columns: `zone` (= state name), `method`, `target`, `predicted_aggregate`, `abs_error`. | **Fair** stress-test: “If we didn’t know this state’s label, how wrong would we be?” |
| **`hierarchical_validation.csv`** | **Train coarse → predict fine**: e.g. train on **6 zones**, evaluate **37 states** or **state×urban/rural**. Rows for **`reconciled=False`** (raw) and **`True`**. | Compare **MAE**, **Pearson *r*** across rows. | Shows whether **coarser** supervision still gives sensible **finer** patterns. |
| **`hierarchical_validation_detail.csv`** | Finer-grained rows for the same experiments. | Same metrics at more breakdown levels. | Debugging / appendix tables. |
| **`two_level_cv.csv`** | Two-level cross-validation metrics where configured. | As documented in run logs. | Extra robustness checks. |

### 3.3 Per-state / per-admin error tables (`admin_detail_*.csv`)

**Pattern:** `admin_detail_<method>.csv` and often `admin_detail_<method>_depth.csv` for depth targets.

| What it shows | How to read it | Why it’s useful |
|----------------|----------------|-----------------|
| **Errors per state** (or admin unit) for that **method**: predicted vs truth, signed error. | Sort by absolute error to find **worst states**. | **Diagnostics** — where Ridge fails (e.g. Lagos, heterogeneous states). |

Methods include **`ridge`**, **`rwi`**, **`uniform`**, **`heuristic`**, **`gbm`**, **`gam`**, **`wsnn`** etc., depending on what ran.

### 3.4 External and auxiliary validation

| File | What it shows | How to read it | Why it’s useful |
|------|----------------|----------------|-----------------|
| **`dhs_gps_validation.csv`** / **`dhs_gps_validation.txt`** | Model vs **DHS 2018 cluster** deprivation index at GPS points (Spearman, MAE). | **Not** MICS truth — different construct; **positive ρ** = spatial alignment. | **Independent** survey check. |
| **`dhs_mics_crossvalidation.csv`** / **`.txt`** | Comparison between **DHS** and **MICS** constructs. | Read metrics as correlation / MAE notes in file header. | Cross-survey sanity. |
| **`nbs_model_comparison.csv`** / **`.txt`** | Comparison to **NBS** monetary / poverty aggregates where wired. | As column headers indicate. | Linking to **official** monetary poverty layer. |

### 3.5 Explainability and dimension summaries

| File | What it shows | How to read it | Why it’s useful |
|------|----------------|----------------|-----------------|
| **`ridge_feature_contribution_breakdown.csv`** | **Global** Ridge coefficients / grouped **themes** (not geographic). | Which features push predictions up/down nationally. | **Stakeholder** explanation of drivers. |
| **`nga_dimension_summary.csv`** | **Per-dimension** Ridge: alphas, national target vs pred means, optional zone *r*. | **`pred_mean`** is **after reconciliation**. | Quick **dimension model** health check. |
| **`gbm_feature_importances.csv`** | GBM **importances** if GBM ran. | Standard tree model ranking. | Nonlinear counterpart to Ridge themes. |
| **`rwi_uncertainty_analysis.csv`** | Diagnostics around **RWI** baseline and uncertainty. | As columns indicate. | Baseline uncertainty studies. |

### 3.6 Statistical tests and sweeps

| File | What it shows | How to read it | Why it’s useful |
|------|----------------|----------------|-----------------|
| **`significance_tests.csv`** | Paired tests between methods where implemented. | *p*-values / differences. | **Formal** comparison statements (with caution). |
| **`dhs_aux_stack_sweep.csv`**, **`dhs_soft_label_sweep.csv`** | Ablation / hyperparameter **sweeps** for DHS-assisted Ridge. | Each row = setting; compare metrics columns. | Choosing **`dhs_aux_*_scale`** in config. |

---

## 4. Interim “tables” (`Data/interim/nga/`)

Not final deliverables, but **inputs** to the next step; useful when debugging.

| File | Meaning |
|------|---------|
| **`nga_targets*.csv`** | MICS (and merged) **targets** at national / zone / state / urban-rural levels. |
| **`nga_dimension_targets.csv`** | **Kyriaki** dimension targets per state (+ DHS HAZ merge for nutrition when used). |
| **`nga_modeling_table.parquet`** | Full **feature matrix + targets** for training. |
| **`nga_dhs_haz_targets.csv`** | Cached **DHS HAZ** nutrition targets by state. |

---

## 5. Quick reference — “I want to…”

| Goal | Start here |
|------|------------|
| Show the **composite** map in a meeting | **`nga_predictions_map_sample.html`** or **`nga_comparison_map.html`** |
| Explain **percentages** on dimension comparison map | Open **`nga_dimension_comparison_map.html`** — read the **“How to read this map”** section at top |
| Analyse cells in Python | **`nga_predictions.parquet`** + **`nga_full_consolidated.parquet`** |
| LGAs for reporting | **`nga_lga_predictions.csv`** |
| Quote **model accuracy** vs states | **`evaluation_summary.csv`** |
| Quote **generalisation** without reconciliation bias | **`lozo_evaluation.csv`** (Ridge), **`hierarchical_validation.csv`** (`reconciled=False`) |
| External validation | **`dhs_gps_validation.txt`** |
| Per-state mistakes | **`admin_detail_ridge.csv`** |

---

*Aligned with pipeline outputs under `Data/outputs/nga/`. Regenerate by running `python main.py --country nga` and, for dimensions, `python src/scripts/run_dimension_models.py --country nga`.*
