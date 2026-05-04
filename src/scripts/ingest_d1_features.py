"""
D1 External Features Orchestrator — Nigeria.

Runs all three D1 ingestion scripts, assembles a merged state-level feature
table, and joins it into the modeling table (`nga_modeling_table.parquet`).

Steps:
  1. ingest_iiag    → national-level governance scalars (broadcast to all cells)
  2. ingest_nbs_mpi → state-level NBS MPI survey features
  3. ingest_nemis   → state-level NEMIS school system features
  4. Merge all into {modeling_table} by `subregion` (state name)
  5. Persist enriched modeling table and d1_features parquet
  6. Print new feature list for config_nga.yaml

Usage
-----
  python src/scripts/ingest_d1_features.py --country nga [--dry-run]

After running, add the printed feature names under `modeling.features` in
config/config_nga.yaml, then re-run the full pipeline:
  python main.py --country nga
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.scripts.ingest_iiag    import load_iiag, IIAG_XLSX
from src.scripts.ingest_nbs_mpi import (
    load_section_a, load_section_j, load_section_i,
    load_section_e, load_section_f, aggregate_to_state,
    MPI_DIR,
)
from src.scripts.ingest_nemis   import load_level, aggregate_level, NEMIS_DIR

log = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────────
D1_OUT_DIR = ROOT / "Data/Nigeria/d1_external"
D1_FEATURES_PARQUET = D1_OUT_DIR / "nga_d1_features.parquet"
D1_FEATURES_CSV     = D1_OUT_DIR / "nga_d1_features.csv"


# ── IIAG: national indicator → scalar value (broadcast to all cells) ──────────
IIAG_INDICATORS = [
    ("iiag_overall_governance",    "GOVERNANCE"),
    ("iiag_human_development",     "HD"),
    ("iiag_health",                "HEALTH"),
    ("iiag_education",             "EDUC"),
    ("iiag_soc_protection",        "SOCPROT"),
    ("iiag_abs_lived_poverty",     "AbsLivPov"),
    ("iiag_child_maternal_health", "ContChildMatHealth"),
]

NEMIS_LEVELS = {
    "pre":     "PRE-PRIMARY.xlsx",
    "primary": "PRIMARY.xlsx",
    "jss":     "JSS.xlsx",
    "sss":     "SSS.xlsx",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_iiag_scalars() -> dict:
    """Return {col: value} for Nigeria's latest IIAG year."""
    if not IIAG_XLSX.exists():
        log.warning("IIAG file not found — skipping IIAG features")
        return {}
    df = load_iiag(IIAG_XLSX)
    latest = df[df["year"] == df["year"].max()].iloc[0]
    return {col: float(latest.get(col, np.nan)) for col, _ in IIAG_INDICATORS}


def _build_nbs_state() -> pd.DataFrame:
    """Return state-level NBS MPI features DataFrame."""
    if not MPI_DIR.exists():
        log.warning("NBS MPI survey folder not found — skipping NBS features")
        return pd.DataFrame(columns=["subregion"])
    log.info("Building NBS MPI state features …")
    weights = load_section_a()
    j_df = load_section_j(weights)
    i_df = load_section_i(weights)
    e_df = load_section_e(weights)
    f_df = load_section_f(weights)
    return aggregate_to_state(weights, j_df, i_df, e_df, f_df)


def _build_nemis_state() -> pd.DataFrame:
    """Return state-level NEMIS school features DataFrame."""
    agg_dfs = []
    for prefix, fname in NEMIS_LEVELS.items():
        path = NEMIS_DIR / fname
        if not path.exists():
            log.warning("NEMIS file not found: %s — skipping level %s", path, prefix)
            continue
        df = load_level(fname)
        if df is None:
            continue
        agg = aggregate_level(df, prefix)
        agg_dfs.append(agg)

    if not agg_dfs:
        log.warning("No NEMIS files loaded — returning empty DataFrame")
        return pd.DataFrame(columns=["subregion"])

    merged = agg_dfs[0]
    for other in agg_dfs[1:]:
        merged = merged.merge(other, on="subregion", how="outer")

    school_count_cols = [c for c in merged.columns if c.endswith("_schools")]
    enrol_cols        = [c for c in merged.columns if c.endswith("_enrol") and "pct" not in c]
    merged["nemis_total_schools"] = merged[school_count_cols].fillna(0).sum(axis=1)
    merged["nemis_total_enrol"]   = merged[enrol_cols].fillna(0).sum(axis=1)
    pub_cols = [c for c in merged.columns if "public_pct" in c]
    rur_cols = [c for c in merged.columns if "rural_pct"  in c]
    if pub_cols:
        merged["nemis_public_pct"] = merged[pub_cols].mean(axis=1)
    if rur_cols:
        merged["nemis_rural_pct"]  = merged[rur_cols].mean(axis=1)

    return merged


def build_d1_state_features() -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      state_df : DataFrame indexed by subregion (state name) with all D1 columns
      scalars  : dict of {col: value} for national-level IIAG constants
    """
    log.info("── IIAG governance scalars ──")
    scalars = _build_iiag_scalars()
    if scalars:
        log.info("  Extracted %d IIAG scalar features (latest year)", len(scalars))
        for k, v in scalars.items():
            log.info("    %s = %.1f", k, v)

    log.info("── NBS MPI state features ──")
    nbs_df = _build_nbs_state()

    log.info("── NEMIS school features ──")
    nemis_df = _build_nemis_state()

    # Merge state-level frames
    if nbs_df.empty and nemis_df.empty:
        log.warning("No state-level D1 features available")
        return pd.DataFrame(columns=["subregion"]), scalars

    if nbs_df.empty:
        state_df = nemis_df
    elif nemis_df.empty:
        state_df = nbs_df
    else:
        state_df = nbs_df.merge(nemis_df, on="subregion", how="outer")

    state_df = state_df.sort_values("subregion").reset_index(drop=True)
    return state_df, scalars


def merge_into_modeling_table(
    modeling_table_path: Path,
    state_df: pd.DataFrame,
    scalars: dict,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Join D1 features into the modeling table:
      - State-level features joined via `subregion`
      - IIAG scalars broadcast to every row

    Returns the enriched DataFrame without writing (if dry_run=True).
    """
    log.info("Loading modeling table: %s", modeling_table_path)
    mt = pd.read_parquet(modeling_table_path)
    n_before = len(mt.columns)

    # Drop any existing D1 columns to avoid duplicates on re-runs
    existing_d1 = [c for c in mt.columns if c.startswith(("nbs_", "nemis_", "iiag_"))]
    if existing_d1:
        log.info("Dropping %d pre-existing D1 columns for refresh", len(existing_d1))
        mt = mt.drop(columns=existing_d1)

    # 1. State-level join
    if not state_df.empty and "subregion" in state_df.columns:
        feat_cols = [c for c in state_df.columns if c != "subregion"]
        mt = mt.merge(state_df[["subregion"] + feat_cols], on="subregion", how="left")
        log.info("  Joined %d state-level D1 feature columns", len(feat_cols))

    # 2. Broadcast IIAG scalars
    for col, val in scalars.items():
        mt[col] = val
    if scalars:
        log.info("  Broadcast %d IIAG scalar columns to all %d cells", len(scalars), len(mt))

    n_new = len(mt.columns) - n_before
    log.info("  Modeling table: %d cells, +%d columns → %d total",
             len(mt), n_new, len(mt.columns))

    if not dry_run:
        mt.to_parquet(modeling_table_path, index=False)
        log.info("  Saved enriched modeling table → %s", modeling_table_path)

    return mt


def print_config_snippet(new_cols: list) -> None:
    """Print YAML snippet ready to paste into config_nga.yaml."""
    print("\n" + "=" * 70)
    print("Add these feature names under `modeling.features` in config_nga.yaml:")
    print("=" * 70)
    for col in sorted(new_cols):
        print(f'    - "{col}"')
    print("=" * 70)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Ingest D1 external features for Nigeria")
    parser.add_argument("--country", default="nga", help="Country code (default: nga)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build features but do not write to modeling table")
    args = parser.parse_args()

    # Load config to get modeling table path
    from src.utils.config_loader import load_config
    cfg_path = ROOT / f"config/config_{args.country}.yaml"
    cfg = load_config(str(cfg_path))
    modeling_table_path = Path(cfg["paths"]["modeling_table_file"])

    log.info("=== D1 External Feature Ingestion — %s ===", args.country.upper())

    # 1. Build features
    state_df, scalars = build_d1_state_features()

    # 2. Save intermediate D1 features file
    D1_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not state_df.empty:
        state_df.to_parquet(D1_FEATURES_PARQUET, index=False)
        state_df.to_csv(D1_FEATURES_CSV, index=False)
        log.info("Saved D1 state features → %s", D1_FEATURES_PARQUET)
    else:
        log.warning("No state-level D1 features to save")

    # 3. Merge into modeling table
    new_cols = []
    if modeling_table_path.exists():
        mt = merge_into_modeling_table(modeling_table_path, state_df, scalars,
                                       dry_run=args.dry_run)
        new_cols = [c for c in mt.columns
                    if c.startswith(("nbs_", "nemis_", "iiag_"))]
    else:
        log.warning("Modeling table not found at %s — skipping join", modeling_table_path)
        new_cols = (
            [c for c in state_df.columns if c != "subregion"]
            + list(scalars.keys())
        )

    # 4. Print config snippet
    print_config_snippet(new_cols)

    if args.dry_run:
        print("\n[DRY RUN] Modeling table NOT modified.")
    else:
        print(f"\nDone. {len(new_cols)} D1 feature columns added to modeling table.")
        print("Next step: add the feature names above to config_nga.yaml, then run:")
        print("  python main.py --country nga")


if __name__ == "__main__":
    main()
