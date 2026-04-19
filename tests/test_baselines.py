"""Tests for baseline methods — all use synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.baselines.uniform import apply_uniform_baseline


def _make_df():
    rng = np.random.default_rng(42)
    zones = ["Urban"] * 30 + ["Rural"] * 30 + ["KMA"] * 30
    targets = {"Urban": 22.92, "Rural": 34.65, "KMA": 34.57}
    return pd.DataFrame({
        "subregion": zones,
        "population": rng.uniform(10, 1000, 90),
        "rwi": rng.normal(0, 1, 90),
        "moderate_prevalence": [targets[z] for z in zones],
        "severe_prevalence": [targets[z] * 0.5 for z in zones],
        "moderate_depth": [targets[z] * 0.3 for z in zones],
        "severe_depth": [targets[z] * 0.15 for z in zones],
        "in_modeling_sample": [True] * 90,
    })


class TestUniformBaseline:
    def test_assigns_zone_target(self):
        df = _make_df()
        result = apply_uniform_baseline(df)
        for zone in ["Urban", "Rural", "KMA"]:
            zmask = result["subregion"] == zone
            target = result.loc[zmask, "moderate_prevalence"].iloc[0]
            uniform = result.loc[zmask, "uniform_moderate"].values
            assert np.allclose(uniform, target)

    def test_no_nan_in_valid_zones(self):
        df = _make_df()
        result = apply_uniform_baseline(df)
        valid = result["subregion"].isin(["Urban", "Rural", "KMA"])
        assert result.loc[valid, "uniform_moderate"].notna().all()


class TestRWIBaseline:
    def test_preserves_ordering(self):
        """RWI baseline should assign higher poverty to lower-RWI cells."""
        from src.baselines.rwi_redistribution import apply_rwi_baseline

        df = _make_df()
        result = apply_rwi_baseline(df)

        for zone in ["Urban", "Rural", "KMA"]:
            zmask = result["subregion"] == zone
            rwi = result.loc[zmask, "rwi"].values
            pred = result.loc[zmask, "rwi_moderate"].values
            # Spearman r between rwi and predictions should be negative
            from scipy.stats import spearmanr
            r, _ = spearmanr(rwi, pred)
            assert r < 0, f"Zone {zone}: RWI baseline should be negatively correlated with RWI"


class TestHeuristicBaseline:
    def test_tercile_stratification(self):
        """Heuristic baseline should create variation based on RWI terciles."""
        from src.baselines.heuristic import _apply_tercile_heuristic

        df = _make_df()
        
        # Test one zone
        zone_mask = df["subregion"] == "Urban"
        scores = _apply_tercile_heuristic(
            df, zone_mask, "moderate_prevalence"
        )
        
        # Should have variation (not all identical)
        assert scores.std() > 0, "Heuristic should create spatial variation"
        
        # Bottom tercile should have higher poverty on average than top tercile
        zone_df = df[zone_mask].copy()
        rwi_terciles = pd.qcut(
            zone_df["rwi"],
            q=3,
            labels=['bottom', 'middle', 'top'],
            duplicates='drop'
        )
        
        bottom_scores = scores[rwi_terciles == 'bottom']
        top_scores = scores[rwi_terciles == 'top']
        
        # Bottom tercile (poorest) should have higher poverty
        assert bottom_scores.mean() > top_scores.mean(), \
            "Poorest tercile should have higher poverty than richest tercile"
    
    def test_full_pipeline_run(self):
        """Test full heuristic baseline with all targets."""
        from src.baselines.heuristic import _apply_tercile_heuristic
        
        df = _make_df()
        
        # Test that all required targets can be processed
        for target_col in ["moderate_prevalence", "severe_prevalence", 
                          "moderate_depth", "severe_depth"]:
            for zone in ["Urban", "Rural", "KMA"]:
                zone_mask = df["subregion"] == zone
                scores = _apply_tercile_heuristic(
                    df, zone_mask, target_col
                )
                
                # Should produce valid scores
                assert len(scores) == zone_mask.sum()
                assert not np.any(np.isnan(scores))
                assert np.all(scores >= 0)  # Poverty rates should be non-negative
    
    def test_negative_correlation_with_rwi(self):
        """Heuristic should assign higher poverty to lower-RWI cells."""
        from src.baselines.heuristic import _apply_tercile_heuristic
        
        df = _make_df()
        
        for zone in ["Urban", "Rural", "KMA"]:
            zone_mask = df["subregion"] == zone
            scores = _apply_tercile_heuristic(
                df, zone_mask, "moderate_prevalence"
            )
            
            rwi = df.loc[zone_mask, "rwi"].values
            
            # Should have negative correlation
            from scipy.stats import spearmanr
            r, _ = spearmanr(rwi, scores)
            assert r < 0, f"Zone {zone}: Heuristic should be negatively correlated with RWI"

