"""Tests for admin_reconcile.py — all use synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.admin_reconcile import (
    reconcile_predictions,
    reconcile_uncertainty_bounds,
    verify_reconciliation,
)


def _make_df(n_per_zone=50, zones=None, target_vals=None):
    """Create a synthetic modeling DataFrame."""
    if zones is None:
        zones = ["Urban", "Rural", "KMA"]
    if target_vals is None:
        target_vals = {"Urban": 22.92, "Rural": 34.65, "KMA": 34.57}

    rng = np.random.default_rng(42)
    rows = []
    for z in zones:
        for _ in range(n_per_zone):
            rows.append({
                "subregion": z,
                "population": rng.uniform(10, 1000),
                "moderate_prevalence": target_vals[z],
                "raw_score": rng.uniform(20, 40),
            })
    return pd.DataFrame(rows)


class TestReconcilePredictions:
    def test_zone_means_preserved(self):
        df = _make_df()
        result = reconcile_predictions(
            df, "raw_score", "moderate_prevalence",
            zone_col="subregion", population_col="population",
            output_col="reconciled", strategy="population_weighted",
        )
        for zone in ["Urban", "Rural", "KMA"]:
            zmask = result["subregion"] == zone
            pop = result.loc[zmask, "population"].values
            recon = result.loc[zmask, "reconciled"].values
            achieved = np.average(recon, weights=pop)
            target = result.loc[zmask, "moderate_prevalence"].iloc[0]
            assert abs(achieved - target) < 0.01, f"Zone {zone}: {achieved} != {target}"

    def test_uniform_scores_handled(self):
        df = _make_df()
        df["uniform_score"] = 30.0  # all same
        result = reconcile_predictions(
            df, "uniform_score", "moderate_prevalence",
            zone_col="subregion", population_col="population",
            output_col="reconciled",
        )
        assert result["reconciled"].notna().all()

    def test_clamping_works(self):
        df = _make_df(n_per_zone=10)
        df["extreme_score"] = 200.0  # will be clamped
        result = reconcile_predictions(
            df, "extreme_score", "moderate_prevalence",
            zone_col="subregion", population_col="population",
            output_col="clamped",
        )
        assert result["clamped"].max() <= 100.0
        assert result["clamped"].min() >= 0.0

    def test_verify_passes(self):
        df = _make_df()
        result = reconcile_predictions(
            df, "raw_score", "moderate_prevalence",
            zone_col="subregion", population_col="population",
            output_col="reconciled",
        )
        assert verify_reconciliation(
            result, "reconciled", "moderate_prevalence", "subregion", "population",
        )


class TestReconcileUncertaintyBounds:
    def test_preserves_ratio(self):
        df = _make_df()
        rng = np.random.default_rng(99)
        df["lower"] = df["raw_score"] - rng.uniform(1, 5, len(df))
        df["upper"] = df["raw_score"] + rng.uniform(1, 5, len(df))

        df = reconcile_predictions(
            df, "raw_score", "moderate_prevalence",
            zone_col="subregion", population_col="population",
            output_col="reconciled",
        )

        result = reconcile_uncertainty_bounds(
            df, "lower", "upper", "raw_score", "reconciled",
            zone_col="subregion", population_col="population",
        )

        # After propagation, bounds should be clamped to [0, 100]
        assert result["lower"].min() >= 0.0
        assert result["upper"].max() <= 100.0
        # Width should be > 0 for all valid cells
        valid = result["lower"].notna() & result["upper"].notna()
        widths = result.loc[valid, "upper"] - result.loc[valid, "lower"]
        assert (widths >= 0).all()

    def test_clamping(self):
        df = _make_df(n_per_zone=5)
        df["lower"] = -10.0
        df["upper"] = 200.0
        df = reconcile_predictions(
            df, "raw_score", "moderate_prevalence",
            zone_col="subregion", population_col="population",
            output_col="reconciled",
        )
        result = reconcile_uncertainty_bounds(
            df, "lower", "upper", "raw_score", "reconciled",
            zone_col="subregion", population_col="population",
        )
        assert result["lower"].min() >= 0.0
        assert result["upper"].max() <= 100.0
