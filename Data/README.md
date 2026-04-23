# Data Directory Layout

This repository keeps Nigeria inputs and outputs under `Data/Nigeria` and `Data/outputs/nga`.

## Nigeria raw and processed data

- `Data/Nigeria/dhs/raw/`
  - DHS raw deliveries and GIS folders (`NGKR7BFL`, `NGHR7BFL`, `NGGE7BFL`, SPSS bundles)
- `Data/Nigeria/dhs/`
  - Processed DHS outputs used by the pipeline (`nga_dhs_cluster_deprivation*.csv/.geojson`)
- `Data/Nigeria/features/`
  - Engineered feature datasets (education, health, GHSL, rainfall, etc.)
- `Data/Nigeria/nbs/`
  - NBS poverty reference files
- `Data/Nigeria/lsms/`
  - LSMS validation assets

## Pipeline outputs

- `Data/interim/nga/` — cached intermediate tables
- `Data/outputs/nga/tables/` — prediction tables
- `Data/outputs/nga/eval/` — evaluation metrics and validation reports
- `Data/outputs/nga/maps/` — map-ready geojson/html/png outputs

Keep large/licensed raw files in `Data/Nigeria/dhs/raw/` and avoid adding them to git.
