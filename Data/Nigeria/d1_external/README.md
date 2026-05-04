# D1 external datasets — EMIS / governance / NBS LGA

This folder holds **non-MICS** predictors and validation sources. Files here are **not** wired into `nga_modeling_table.parquet` yet; ingestion is a follow-up task once columns are harmonised to GADM LGA / state.

---

## 1. Nigeria EMIS (NEMIS) — **partially downloaded**

**Official portal:** https://nemis.education.gov.ng/downloads  
**Mirror page (FME):** https://education.gov.ng/educational-data/

### What we have locally (`nemis/`)

Bulk school listings (Excel) were pulled from NEMIS with:

```bash
BASE="https://nemis.education.gov.ng"
curl -skL "$BASE/school/PRE-PRIMARY.xlsx" -o nemis/PRE-PRIMARY.xlsx
curl -skL "$BASE/school/PRIMARY.xlsx"   -o nemis/PRIMARY.xlsx
curl -skL "$BASE/school/JSS.xlsx"      -o nemis/JSS.xlsx
curl -skL "$BASE/school/SSS.xlsx"       -o nemis/SSS.xlsx
```

- `-k` is needed on some machines because the site’s TLS chain may not verify with default CA bundles.
- `PRIMARY.xlsx` includes columns **`STATE`**, **`LGA`**, **`SCHOOL NAME`**, enrolment by grade — suitable for **LGA-level aggregates** after name harmonisation to GADM `NAME_2`.

**Digests / PDFs** (national aggregates, indicators): same downloads page — use a browser for PDFs, e.g. *Nigeria Education Digest 2022*.

**`asc_instruments.zip`:** do **not** use the relative link from the downloads page with `curl` alone; the server returned HTML. Download **Annual School Census** materials through the browser from the same portal if needed.

---

## 2. Mo Ibrahim / IIAG governance — **`2024-IIAG-scores.xlsx` (on repo)**

**Yes — this is the right dataset for the “Mo Ibrahim / IIAG” part of D1.** It is the standard **wide** IIAG export: **one row per country and year**, hundreds of governance indicator columns (overall score, Security & Rule of Law, etc.). **There is no LGA breakdown** in IIAG; Nigeria is one row per year. Use it as **national context** (scalar features or documentation), not for within-Nigeria LGA disaggregation.

**Path:** project root `2024-IIAG-scores.xlsx` (sheet `Sheet1`). Rows 0–5 are metadata; **data start around row 6** with `Country`, `Year`, then scores.

**Updates / other years:** https://iiag.online/downloads.html (browser; Cloudflare may block `curl`).

---

## 3. NBS MPI microdata — **`Nigeria Multidimensional Poverty Index Survey/` (on repo)**

**Yes — this is exactly what we meant by “NBS MPI microdata” for LGA-linked poverty / deprivation.** The folder of Stata **`.dta`** section files is the **Nigeria Multidimensional Poverty Index Survey** microdata (household + modules). For the pipeline, the critical file is:

| File | Role |
| :--- | :--- |
| **`SECTION A _ IDENTIFICATION.dta`** | `hh_id`, survey weights (`hh_wgt`, `pop_wgt`, …), and **geographic fields** (`a1`, `a2`, … — state / LGA / EA per NBS documentation). Use these to **aggregate to LGA** and join to GADM `NAME_2` after harmonising labels. |
| Other `SECTION *.dta` | Health, housing, food security, etc. — optional inputs if you derive **LGA-level summary statistics** as extra predictors or validation targets. |

**Path:** repo root folder `Nigeria Multidimensional Poverty Index Survey/` (21 section `.dta` files).

**Catalogue:** https://microdata.nigerianstat.gov.ng/index.php/catalog/71  

**Licence / Git:** If your NBS access terms **forbid redistribution**, keep `.dta` files **local** or in a **private** bucket and add the folder to `.gitignore` for public repos.

**Aggregated tables:** https://www.nigerianstat.gov.ng/ — published PDF/Excel as available.

---

## 4. National proxies (API) — `governance/worldbank_nga_primary_enrollment.json`

**World Bank API** (no key, national time series — **not** LGA):

```text
https://api.worldbank.org/v2/country/NGA/indicator/SE.PRM.ENRR?format=json&per_page=100
```

Saved as JSON for convenience. Useful as a **single-country trend** control, not for disaggregation.

---

## Next step for the pipeline

1. Build a script that reads `nemis/*.xlsx`, normalises `STATE` / `LGA` strings to GADM, aggregates counts → `nga_emis_lga_features.csv`.  
2. Join that parquet/CSV into the grid feature build (same pattern as MICS state aggregates).  
3. Register new column names under `modeling.features` in `config_nga.yaml`.
