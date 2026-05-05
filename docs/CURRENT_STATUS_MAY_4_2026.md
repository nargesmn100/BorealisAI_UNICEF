# Current status — May 4, 2026

> **Research tool only.** Outputs are not official poverty statistics.  
> UNICEF × RBC Borealis AI — Nigeria child deprivation pipeline.

This snapshot summarises the repository after a **fresh Nigeria pipeline run on 4 May 2026** (~21:04 local). The living master checklist remains **`PROJECT_STATUS.md`**. Operational detail: **`HOW_TO_RUN.md`**.

---

## Pipeline rerun (recorded)

**Command executed**

```bash
python main.py --country nga --skip-gbm --skip-gam --skip-wsnn
```

Gradient boosting, GAM, and WSNN were skipped for runtime; **Ridge** (plus baselines and hierarchical checks that still instantiate GBM where the evaluator does so) produced refreshed **`Data/outputs/nga/`** tables, maps, and eval CSVs. Exit code: **0**.

**Artifacts touched** (among others): `nga_predictions.parquet`, `evaluation_summary.csv`, `lozo_evaluation.csv`, `hierarchical_validation.csv`, `dhs_gps_validation.txt`, Folium maps, LGA CSV/GeoJSON.

---

## Executive summary — Nigeria findings

- **103,424** grid cells (~**2.4 km** spacing, RWI-aligned).
- **Composite model** is trained on **MICS6-derived state targets** for **multidimensional child deprivation** (moderate / severe prevalence and depth), with **admin reconciliation** so population-weighted state aggregates match survey totals.
- **Kyriaki dimension models** (shelter, sanitation, water, nutrition, education bands, health bands) are **separate** Ridge models with **their own** state-level targets; they are **not** simple additive pieces of the composite score (see **§6**).
- **Leakage-aware feature set:** **38** predictors (22 geospatial + 16 D1); eight overlapping MICS survey rates removed.
- **Nutrition dimension targets** use **DHS 2018 HAZ stunting** when `NGKR7BFL` + `NGGE7BFL` are present; otherwise MDD proxy.

---

## Accuracy — composite “poverty / deprivation score” (state level)

These numbers describe **how well each method matches MICS state targets** after the pipeline’s reconciliation logic (see `evaluation_summary.csv`). “Accuracy” here means **agreement with coarse survey labels**, not proof of cell-level truth.

### Headline metrics (latest run)

| Method | Pearson *r* vs state targets | Spearman *r* | Notes |
|--------|-------------------------------|--------------|--------|
| **RWI redistribution** | **0.794** | **0.800** | Strong ranking vs states using wealth proxy only. |
| **Heuristic (urban/rural)** | 0.674 | 0.733 | Middle ground. |
| **Ridge (learned)** | **0.398** | **0.313** | Interpretable model + DHS aux; lower *r* after leakage removal (honest vs inflated). |
| **Uniform** | 0.314 | 0.280 | Baseline spread of national/zone rates. |

**Depth targets** (severity layer): Ridge Pearson *r* ≈ **0.29** for `ridge_depth` vs `moderate_depth` targets in the same summary file — moderate deprivation is modelled more reliably than depth in this weak-supervision setting.

### Held-out geography — LOZO-style zone errors

File: **`Data/outputs/nga/eval/lozo_evaluation.csv`** (leave-one-**state** (**`subregion`**) out: train on all other states, compare **raw** Ridge aggregate on held-out cells vs target — **no** reconciliation on the hold-out).

| Method | Mean absolute error (pp) | Median AE (pp) | Worst case (pp) |
|--------|--------------------------|----------------|------------------|
| Uniform / RWI | **15.63** | 15.70 | 29.05 |
| **Ridge** | **12.70** | 9.10 | **76.36** (Lagos — high-density outlier) |

Ridge improves on the naive redistributed baselines on average, but **fails badly on a few unusual states** when zones are held out.

### Finer vs coarser labels — hierarchical validation

File: **`Data/outputs/nga/eval/hierarchical_validation.csv`**.

| Train level | Predict | Model | MAE (pp) | Pearson *r* | Interpretation |
|-------------|---------|-------|----------|-------------|----------------|
| 6 **zones** | **37 states** | Ridge (reconciled) | **7.53** | **0.846** | Coarse training still recovers **state ranking** well. |
| 6 zones | state × urban/rural | Ridge (reconciled) | **15.20** | **0.512** | Sub-state split is **much harder** when trained only from zones. |
| **37 states** | state × urban/rural | Ridge (reconciled) | **13.71** | **0.576** | Best sub-state result when state labels are available. |

**Takeaway:** the model is **relatively strong at state-level ranking** when trained with zone-level signal (or full state training), but **weaker at urban/rural splits** — that is the main “granularity gap.”

### External check — DHS 2018 GPS clusters

File: **`Data/outputs/nga/eval/dhs_gps_validation.txt`** (after this run).

| Metric | Ridge | RWI |
|--------|------|-----|
| Spearman ρ vs DHS cluster deprivation index | **0.600** | 0.542 |
| Pearson *r* | **0.584** | — |
| MAE | **14.45 pp** (cross-metric) | — |

**1382** valid cluster points; mean distance from DHS coordinate to grid centre **1.02 km**. The DHS index is **not** the same as MICS moderate prevalence — positive correlation means **spatial co-movement**, not calibration to MICS.

---

## Accuracy — per-dimension models (Kyriaki)

File: **`Data/outputs/nga/eval/nga_dimension_summary.csv`** (last produced when **`run_dimension_models.py`** was run; main.py does not regenerate this).

National means from that table (36-state coverage where noted):

| Dimension | Target mean % | Predicted mean % |
|-----------|-----------------|------------------|
| Shelter | 56.06 | 52.01 |
| Sanitation | 1.98 | 1.82 |
| Water | 4.25 | 4.28 |
| Nutrition | 41.01 | 44.56 |
| Education 5–14 | 33.60 | 38.35 |
| Education 15–17 | 37.35 | 45.74 |
| Health 12–35m | 89.22 | 90.23 |
| Health 36–59m | 22.78 | 22.77 |

Nutrition targets correspond to **DHS HAZ stunting** (label in `nga_dimension_summary.csv` updated to “Nutrition (DHS HAZ stunting)”).

---

## Do dimension scores “sum up” to the coarser composite score?

**No — not as a sum, and not automatically across scales.**

1. **Composite MICS target (what `main.py` trains on)**  
   Defined in **`src/targets/compute_mics_deprivation.py`**: a child is **moderately** multidimensionally deprived if they are deprived in **≥ 2 of six UNICEF dimensions** — nutrition, health, education, WASH, housing, information — each with its own rule set. That produces a **single prevalence** per state. It is **not** the sum of six dimension percentages (which would count overlapping children many times).

2. **Kyriaki dimension targets (what `run_dimension_models.py` trains on)**  
   Use **different** definitions and age bands (e.g. shelter vs “housing” in the six, split education, split health, nutrition from **DHS HAZ** when available). They answer **different scientific questions** than the six-dimension UNICEF aggregate.

3. **Reconciliation**  
   The **composite** Ridge model is reconciled to **composite** state totals. Each **dimension** Ridge model is reconciled to **that dimension’s** state totals. There is **no** mathematical constraint that cell-level dimension predictions add up to the composite cell prediction.

4. **Large-area aggregates**  
   If you population-weight **dimension** predictions up to state level, you recover each **dimension’s** target family by construction (after reconciliation). You **do not** recover the **composite** prevalence unless you rebuild the composite definition from child-level joint deprivation flags — which these separate models do not jointly estimate.

**Practical implication:** treat **composite maps** as the headline deprivation layer consistent with UNICEF-style multidimensional poverty at state scale; treat **dimension maps** as **thematic deep dives** that do not need to sum to the composite.

---

## Reconciliation vs “raw” accuracy — short pointer

Survey truth exists only for **coarse** areas (states). The model predicts **cells**, then **reconciles** within each state so population-weighted means match MICS (composite or per-dimension targets). That fixes totals but mixes **learned spatial pattern** with a **scaling step**.

**Where to read accuracy without reconciliation:** **`lozo_evaluation.csv`** (Ridge = raw aggregate on held-out state), **`hierarchical_validation.csv`** rows with **`reconciled=False`**, and **`dhs_gps_validation.txt`** (external DHS clusters). **`evaluation_summary.csv`** Pearson *r* for Ridge is dominated by reconciled agreement — use LOZO / hierarchical raw / DHS for generalisation of the mapping.

Full explanation: **`HOW_TO_RUN.md` §7**.

---

## Maps for demonstrations (which `.html` to open)

Prioritised list — paths under `Data/outputs/nga/maps/`:

| Order | File | Role |
|:-----:|------|------|
| 1 | **`nga_comparison_map.html`** | Best **default for slides**: LGA polygons, truth vs model. |
| 2 | **`nga_predictions_map_sample.html`** | Composite grid — sampled/clustered, usable live. |
| 3 | **`nga_dimension_comparison_map.html`** | Multi-panel **education / shelter / nutrition / …** in one browser tab. |
| 4 | **`nga_uncertainty_map.html`** | When discussing uncertainty / CI width. |
| 5 | **`nga_dimension_<name>_map.html`** | Single-dimension deep dive. |
| 6 | **`nga_predictions_map.html`** | Full grid only if needed — huge and slow. |

Details and caveats: **`HOW_TO_RUN.md` §5.1**.

---

## Features and integrity

| Item | Detail |
|------|--------|
| Training features | **38** (22 non-survey geospatial + 16 D1 NBS/NEMIS; IIAG stored but not used as varying features). |
| Leakage | Eight MICS-derived rates removed from base features (May 2026). |
| DHS auxiliary Ridge | Stacked cluster loss active (`dhs_aux_*_scale` in `config_nga.yaml`). |

---

## Recommended next steps

1. Optional full model run: `python main.py --country nga` (include GBM/GAM/WSNN) for complete method comparison.
2. Regenerate **dimension Folium maps** (if needed): `python src/scripts/run_dimension_models.py --country nga` (omit `--no-maps`). Summary CSV was refreshed the same day as this snapshot.
3. **`TECHNICAL_OVERVIEW.md`** — align grid wording (~2.4 km vs informal “1 km²”).
4. **`python src/scripts/dhs_aux_sweep.py`** without `--skip-lozo` if long-run validation of aux weights is needed.
5. **NEMIS / NBS LGA harmonisation** — current NBS and NEMIS joins are at *state* level; sub-state variation requires matching LGA strings to GADM ADM2 (`NAME_2`).
6. **Second country (D2)** — create `config_{iso}.yaml`, supply MICS SPSS + RWI + GADM, run `python main.py --country {iso}`.

---

## Repository structure (after May 4, 2026 cleanup)

| Location | Contents |
|---|---|
| `Data/Nigeria/d1_external/governance/` | `2024-IIAG-scores.xlsx` + processed CSVs |
| `Data/Nigeria/d1_external/nbs_mpi/survey/` | NBS MPI raw `.dta` files (gitignored) |
| `Data/Nigeria/d1_external/nemis/` | NEMIS raw xlsx + `nga_nemis_state.csv` |
| `Data/outputs/nga/{tables,maps,eval}/` | All Nigeria pipeline outputs |
| `docs/` | Reference PDFs (gitignored) |
| `config/config_nga.yaml` | Main Nigeria model config |

---

## Where to read more

| Document | Purpose |
|----------|---------|
| **`HOW_TO_RUN.md`** | Commands, output folders, column meanings, **§5.1 demo HTML**, **§7 reconciliation vs raw** |
| **`OUTPUTS_GUIDE_NGA.md`** | **Every Nigeria map + table**: what it shows, how to read it, why it’s useful |
| **`PROJECT_STATUS.md`** | Full backlog and history |
| **`TECHNICAL_OVERVIEW.md`** | Architecture and data dictionary |
| **`config/config_nga.yaml`** | Paths and model knobs |

---

*End of snapshot — May 4, 2026 (pipeline rerun + metrics documented).*
