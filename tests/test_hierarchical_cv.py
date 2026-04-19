"""
Tests for hierarchical cross-level validation.

Uses synthetic data so no real MICS data is required.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.admin_mappings import (
    NIGERIA_GEOPOLITICAL_ZONES,
    add_geopolitical_zones,
    add_state_urban_rural,
)
from src.targets.compute_mics_deprivation import (
    compute_multilevel_targets,
    _weighted_aggregate,
)
from src.evaluation.hierarchical_cv import (
    _pop_weighted_aggregate,
    _metrics,
    hierarchical_validation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_child_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic child-level deprivation flags for testing."""
    rng = np.random.RandomState(seed)
    states = ["Kano", "Lagos", "Abia", "Borno", "Rivers"]
    return pd.DataFrame({
        "HH1": rng.randint(1, 50, n),
        "HH2": rng.randint(1, 10, n),
        "HH7": rng.choice(range(1, 6), n),   # integer state codes
        "HH6": rng.choice([1, 2], n),          # 1=urban, 2=rural
        "chweight": rng.uniform(0.5, 2.0, n),
        "moderate_deprived": rng.randint(0, 2, n),
        "severe_deprived": rng.randint(0, 2, n),
        "n_deprivations": rng.randint(0, 6, n),
    })


STATE_LABELS = {1: "Kano", 2: "Lagos", 3: "Abia", 4: "Borno", 5: "Rivers"}


def _make_modeling_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic modeling table for hierarchical CV tests."""
    rng = np.random.RandomState(seed)
    states = list(NIGERIA_GEOPOLITICAL_ZONES.keys())[:10]  # use 10 states
    subregions = rng.choice(states, n)
    geo_zones = [NIGERIA_GEOPOLITICAL_ZONES.get(s, "Unknown") for s in subregions]
    is_urban = rng.randint(0, 2, n)
    moderate = rng.uniform(20, 60, n)

    df = pd.DataFrame({
        "cell_id": range(n),
        "latitude": rng.uniform(4, 14, n),
        "longitude": rng.uniform(3, 15, n),
        "rwi": rng.normal(0, 1, n),
        "population": rng.uniform(100, 5000, n),
        "log_population": np.log1p(rng.uniform(100, 5000, n)),
        "smod_class": rng.randint(10, 31, n),
        "is_urban": is_urban,
        "travel_time_cities": rng.uniform(0, 300, n),
        "travel_time_50k": rng.uniform(0, 200, n),
        "log_travel_time_cities": rng.uniform(0, 6, n),
        "log_travel_time_50k": rng.uniform(0, 5, n),
        "population_imputed": rng.randint(0, 2, n),
        "subregion": subregions,
        "geopolitical_zone": geo_zones,
        "state_urban_rural": [f"{s}_{'Urban' if u else 'Rural'}"
                              for s, u in zip(subregions, is_urban)],
        "moderate_prevalence": moderate,
        "severe_prevalence": moderate * 0.3,
        "moderate_depth": rng.uniform(1.5, 3.0, n),
        "severe_depth": rng.uniform(2.0, 4.0, n),
        "in_modeling_sample": True,
    })
    return df


# ---------------------------------------------------------------------------
# admin_mappings tests
# ---------------------------------------------------------------------------

class TestAdminMappings:
    def test_all_states_mapped(self):
        """Every state in the dict should map to one of 6 zones."""
        zones = set(NIGERIA_GEOPOLITICAL_ZONES.values())
        assert zones == {"South East", "South South", "South West",
                         "North Central", "North East", "North West"}

    def test_37_states(self):
        assert len(NIGERIA_GEOPOLITICAL_ZONES) == 37

    def test_add_geopolitical_zones(self):
        df = _make_modeling_df()
        out = add_geopolitical_zones(df, zone_col="subregion")
        assert "geopolitical_zone" in out.columns
        known = out["geopolitical_zone"] != "Unknown"
        assert known.sum() > 0

    def test_add_state_urban_rural(self):
        df = _make_modeling_df()
        out = add_state_urban_rural(df, state_col="subregion", urban_col="is_urban")
        assert "state_urban_rural" in out.columns
        assert out["state_urban_rural"].str.contains("_Urban|_Rural").all()


# ---------------------------------------------------------------------------
# compute_multilevel_targets tests
# ---------------------------------------------------------------------------

class TestComputeMultilevelTargets:
    def test_returns_four_levels(self):
        child_df = _make_child_df()
        result = compute_multilevel_targets(
            child_df, weight_col="chweight", state_col="HH7",
            urban_col="HH6", state_labels=STATE_LABELS,
        )
        assert set(result.keys()) == {"national", "geopolitical_zone", "state", "state_urban_rural"}

    def test_national_single_row(self):
        child_df = _make_child_df()
        result = compute_multilevel_targets(child_df, state_labels=STATE_LABELS)
        assert len(result["national"]) == 1

    def test_state_count(self):
        child_df = _make_child_df()
        result = compute_multilevel_targets(child_df, state_labels=STATE_LABELS)
        # Should have at most 5 states (our synthetic data has 5 state codes)
        assert 1 <= len(result["state"]) <= 5

    def test_prevalence_range(self):
        child_df = _make_child_df()
        result = compute_multilevel_targets(child_df, state_labels=STATE_LABELS)
        for level_df in result.values():
            assert (level_df["moderate_prevalence"] >= 0).all()
            assert (level_df["moderate_prevalence"] <= 100).all()

    def test_national_consistent_with_states(self):
        """National prevalence should be a plausible weighted mean of states."""
        child_df = _make_child_df(n=2000, seed=7)
        result = compute_multilevel_targets(child_df, state_labels=STATE_LABELS)
        nat = result["national"]["moderate_prevalence"].iloc[0]
        state_min = result["state"]["moderate_prevalence"].min()
        state_max = result["state"]["moderate_prevalence"].max()
        assert state_min <= nat <= state_max

    def test_hh6_absent_no_crash(self):
        """If HH6 is absent from child_df, state_urban_rural should be empty or partial."""
        child_df = _make_child_df().drop(columns=["HH6"])
        # Should not raise
        try:
            result = compute_multilevel_targets(child_df, state_labels=STATE_LABELS)
            # state_urban_rural may be empty when HH6 is missing
            assert "state_urban_rural" in result
        except Exception as e:
            pytest.fail(f"Raised unexpectedly: {e}")

    def test_weighted_aggregate_zero_weight(self):
        """_weighted_aggregate with zero total weight returns None."""
        group = pd.DataFrame({
            "chweight": [0.0, 0.0],
            "moderate_deprived": [1, 0],
            "severe_deprived": [0, 0],
            "n_deprivations": [2, 1],
        })
        result = _weighted_aggregate(group, "chweight")
        assert result is None


# ---------------------------------------------------------------------------
# hierarchical_cv metric helpers
# ---------------------------------------------------------------------------

class TestHierarchicalCVHelpers:
    def test_pop_weighted_aggregate_uniform_pop(self):
        preds = np.array([10.0, 20.0, 30.0])
        pops = np.array([1.0, 1.0, 1.0])
        assert _pop_weighted_aggregate(preds, pops) == pytest.approx(20.0)

    def test_pop_weighted_aggregate_skewed_pop(self):
        preds = np.array([10.0, 30.0])
        pops = np.array([3.0, 1.0])  # 3:1 weight toward 10
        result = _pop_weighted_aggregate(preds, pops)
        assert result == pytest.approx(15.0)

    def test_pop_weighted_aggregate_zero_pop(self):
        """Falls back to simple mean when all populations are zero."""
        preds = np.array([10.0, 20.0])
        pops = np.array([0.0, 0.0])
        assert _pop_weighted_aggregate(preds, pops) == pytest.approx(15.0)

    def test_metrics_perfect(self):
        pred = np.array([10.0, 20.0, 30.0])
        actual = np.array([10.0, 20.0, 30.0])
        m = _metrics(pred, actual)
        assert m["mae_pp"] == pytest.approx(0.0)
        assert m["pearson_r"] == pytest.approx(1.0)

    def test_metrics_insufficient_groups(self):
        """With < 3 groups, pearson/spearman should be NaN."""
        pred = np.array([10.0, 20.0])
        actual = np.array([12.0, 18.0])
        m = _metrics(pred, actual)
        assert m["pearson_r"] is None or np.isnan(m["pearson_r"] or float("nan"))


# ---------------------------------------------------------------------------
# hierarchical_validation integration test (synthetic)
# ---------------------------------------------------------------------------

class TestHierarchicalValidationIntegration:
    def _make_cfg(self, tmp_path):
        """Minimal config for hierarchical_validation()."""
        return {
            "modeling": {
                "features": [
                    "rwi", "population", "log_population", "smod_class",
                    "is_urban", "travel_time_cities", "travel_time_50k",
                    "log_travel_time_cities", "log_travel_time_50k",
                ],
                "admin_zone_col": "subregion",
                "ridge": {
                    "alpha_candidates": [0.1, 1.0, 10.0],
                    "cv_folds": 3,
                    "random_state": 42,
                },
            },
            "evaluation": {
                "hierarchical_cv": {
                    "enabled": True,
                    "experiments": [
                        {"train_level": "geopolitical_zone", "eval_level": "state"},
                    ],
                    "models": ["ridge"],
                },
            },
        }

    def _make_eval_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """State-level targets computed from the modeling table directly."""
        rows = []
        for state, grp in df.groupby("subregion"):
            rows.append({
                "group_id": state,
                "moderate_prevalence": grp["moderate_prevalence"].mean(),
            })
        return pd.DataFrame(rows)

    def test_runs_without_error(self, tmp_path):
        df = _make_modeling_df(n=600)
        cfg = self._make_cfg(tmp_path)

        # Write synthetic state-level targets to interim_dir
        eval_targets = self._make_eval_targets(df)
        interim_dir = str(tmp_path / "interim")
        import os
        os.makedirs(interim_dir, exist_ok=True)
        eval_targets.to_csv(
            os.path.join(interim_dir, "nga_targets_state.csv"), index=False
        )

        eval_dir = str(tmp_path / "eval")
        result = hierarchical_validation(df, cfg, interim_dir=interim_dir, eval_dir=eval_dir)
        assert isinstance(result, pd.DataFrame)

    def test_disabled_returns_empty(self, tmp_path):
        df = _make_modeling_df()
        cfg = {
            "modeling": {"features": ["rwi"], "admin_zone_col": "subregion",
                         "ridge": {"alpha_candidates": [1.0], "cv_folds": 2}},
            "evaluation": {"hierarchical_cv": {"enabled": False}},
        }
        result = hierarchical_validation(df, cfg, interim_dir=str(tmp_path), eval_dir=str(tmp_path))
        assert len(result) == 0

    def test_output_columns(self, tmp_path):
        df = _make_modeling_df(n=600)
        cfg = self._make_cfg(tmp_path)
        eval_targets = self._make_eval_targets(df)
        interim_dir = str(tmp_path / "interim")
        import os
        os.makedirs(interim_dir, exist_ok=True)
        eval_targets.to_csv(
            os.path.join(interim_dir, "nga_targets_state.csv"), index=False
        )
        result = hierarchical_validation(df, cfg, interim_dir=interim_dir, eval_dir=str(tmp_path))
        if not result.empty:
            for col in ["train_level", "eval_level", "model", "mae_pp", "n_groups"]:
                assert col in result.columns
