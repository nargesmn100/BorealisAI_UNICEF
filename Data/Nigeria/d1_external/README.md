# D1 external datasets — EMIS / governance / NBS MPI

This folder holds **non-MICS** external predictors and validation sources ingested into the Nigeria pipeline.

```
d1_external/
├── governance/               ← IIAG + World Bank API national indicators
│   ├── 2024-IIAG-scores.xlsx        (source file; large — gitignored)
│   ├── nga_iiag_features.csv        (processed per-year scalars)
│   ├── nga_iiag_latest.csv          (latest year only)
│   └── worldbank_nga_primary_enrollment.json
├── nemis/                    ← NEMIS school listings (xlsx raw + derived)
│   ├── PRE-PRIMARY.xlsx
│   ├── PRIMARY.xlsx
│   ├── JSS.xlsx
│   ├── SSS.xlsx
│   └── nga_nemis_state.csv   (aggregated state-level features)
├── nbs_mpi/                  ← NBS MPI household microdata
│   ├── survey/               ← raw .dta section files (gitignored *.dta)
│   │   └── Nigeria Multidimensional Poverty Index Survey/
│   └── nga_nbs_mpi_state.csv (aggregated state-level features)
├── nga_d1_features.csv       ← combined D1 feature table (state-level)
└── nga_d1_features.parquet
```

---

## 1. Nigeria EMIS (NEMIS) — `nemis/`

**Official portal:** https://nemis.education.gov.ng/downloads  

### What we have locally

Bulk school listings (Excel) — download with:

```bash
BASE="https://nemis.education.gov.ng"
curl -skL "$BASE/school/PRE-PRIMARY.xlsx" -o Data/Nigeria/d1_external/nemis/PRE-PRIMARY.xlsx
curl -skL "$BASE/school/PRIMARY.xlsx"   -o Data/Nigeria/d1_external/nemis/PRIMARY.xlsx
curl -skL "$BASE/school/JSS.xlsx"       -o Data/Nigeria/d1_external/nemis/JSS.xlsx
curl -skL "$BASE/school/SSS.xlsx"       -o Data/Nigeria/d1_external/nemis/SSS.xlsx
```

`PRIMARY.xlsx` includes columns **`STATE`**, **`LGA`**, **`SCHOOL NAME`**, enrolment by grade — suitable for **LGA-level aggregates** after name harmonisation. Ingested by `src/scripts/ingest_nemis.py`.

---

## 2. Mo Ibrahim / IIAG governance — `governance/2024-IIAG-scores.xlsx`

Standard **wide** IIAG export: one row per country and year, hundreds of governance indicator columns. **No LGA breakdown** — Nigeria is one row per year; used as **national-context** scalar features.

**Ingestion script:** `src/scripts/ingest_iiag.py`  
**Updates:** https://iiag.online/downloads.html

---

## 3. NBS MPI microdata — `nbs_mpi/survey/`

The **Nigeria Multidimensional Poverty Index Survey** Stata `.dta` section files (household + modules). Critical files:

| File | Role |
| :--- | :--- |
| `SECTION A _ IDENTIFICATION.dta` | `hh_id`, survey weights, geographic fields (`a1`/`a2` = state/LGA) |
| `SECTION J_HOUSING CHARACTERISTICS_NEW.dta` | floor quality etc. |
| `SECTION I_ WATER AND SANITATION.dta` | WASH indicators |
| `SECTION E_FOOD SECURITY.dta` | HFIAS food security |
| `SECTION F_HEALTH.dta` | health facility access |

**Ingestion script:** `src/scripts/ingest_nbs_mpi.py`  
**Catalogue:** https://microdata.nigerianstat.gov.ng/index.php/catalog/71  
**Note:** `.dta` files are gitignored (restricted licence / large).

---

## 4. National proxy (API) — `governance/worldbank_nga_primary_enrollment.json`

World Bank API (no key, national time series):
```
https://api.worldbank.org/v2/country/NGA/indicator/SE.PRM.ENRR?format=json&per_page=100
```
Useful as single-country trend control, not for disaggregation.

---

## Ingestion pipeline

Run in order (or use the orchestrator):

```bash
python src/scripts/ingest_iiag.py
python src/scripts/ingest_nbs_mpi.py
python src/scripts/ingest_nemis.py
python src/scripts/ingest_d1_features.py   # merges all above into modeling table
python main.py --country nga               # full pipeline
```

## Next steps

- **NEMIS/NBS LGA harmonisation:** join `a2` LGA codes / `LGA` strings to GADM ADM2 (`NAME_2`) — currently state-level only.
- **GBM SHAP:** enable `export_gbm_shap: true` in `config_nga.yaml`.
