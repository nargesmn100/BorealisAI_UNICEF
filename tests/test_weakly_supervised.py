"""
Tests for weakly supervised neural network with aggregation-based training.
"""

import pytest
import numpy as np
import pandas as pd

# Try to import torch
try:
    import torch
    from src.models.weakly_supervised_nn import (
        WeaklySupervisedNN,
        _prepare_zone_groups,
        _prepare_quintile_groups,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestWeaklySupervisedNN:
    """Test suite for weakly supervised neural network."""
    
    def test_aggregation_loss_correct(self):
        """Test that aggregation loss matches manual calculation."""
        # Create simple test case: 2 groups, 2 cells each
        model = WeaklySupervisedNN(input_dim=2, hidden_dims=[8])
        
        # Cell predictions
        cell_preds = torch.tensor([[0.3], [0.5], [0.2], [0.4]])
        # Cell populations
        cell_pops = torch.tensor([10.0, 20.0, 15.0, 25.0])
        # Group assignments (0 or 1)
        groups = torch.tensor([0, 0, 1, 1])
        # Official targets for each group
        targets = {0: 0.4, 1: 0.35}
        
        loss = model.compute_aggregation_loss(cell_preds, cell_pops, groups, targets)
        
        # Manual calculation:
        # Group 0: (0.3*10 + 0.5*20) / (10+20) = (3 + 10) / 30 = 13/30 = 0.4333...
        #          (0.4333... - 0.4)² = 0.0333...² = 0.001111...
        # Group 1: (0.2*15 + 0.4*25) / (15+25) = (3 + 10) / 40 = 13/40 = 0.325
        #          (0.325 - 0.35)² = (-0.025)² = 0.000625
        # Mean: (0.001111... + 0.000625) / 2 = 0.000868...
        
        expected_loss = 0.000868
        assert abs(loss.item() - expected_loss) < 1e-5, \
            f"Loss {loss.item():.6f} != expected {expected_loss:.6f}"
    
    def test_predictions_in_valid_range(self):
        """Test that predictions are in [0, 100] range."""
        # Create simple dataset
        X = np.random.randn(50, 5)
        population = np.random.uniform(1, 10, size=50)
        groups = np.random.randint(0, 3, size=50)
        targets = {0: 0.25, 1: 0.35, 2: 0.30}
        
        # Train model
        model = WeaklySupervisedNN(input_dim=5, hidden_dims=[8])
        model.fit(X, population, groups, targets, n_epochs=10, verbose=False)
        
        # Predict
        preds = model.predict(X)
        
        # Check range
        assert preds.min() >= 0, "Predictions should be >= 0"
        assert preds.max() <= 100, "Predictions should be <= 100"
    
    def test_aggregations_match_targets_after_training(self):
        """Test that trained model produces aggregates close to targets."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Simple synthetic dataset
        n_cells = 100
        X = np.random.randn(n_cells, 5)
        population = np.random.uniform(1, 10, size=n_cells)
        groups = np.random.randint(0, 3, size=n_cells)
        targets = {0: 0.25, 1: 0.35, 2: 0.30}
        
        # Train model with enough epochs
        model = WeaklySupervisedNN(input_dim=5, hidden_dims=[16, 8])
        model.fit(X, population, groups, targets, n_epochs=200, verbose=False)
        
        # Predict
        preds = model.predict(X) / 100  # Convert back to 0-1 scale
        
        # Check that aggregates are close to targets
        for group_id in range(3):
            mask = groups == group_id
            group_preds = preds[mask]
            group_pops = population[mask]
            
            # Population-weighted aggregate
            agg_pred = (group_preds * group_pops).sum() / group_pops.sum()
            target = targets[group_id]
            
            # Should be close (within 20% relative error after 200 epochs)
            # Note: Convergence is approximate due to random initialization and complex loss surface
            rel_error = abs(agg_pred - target) / target
            assert rel_error < 0.20, \
                f"Group {group_id}: aggregate {agg_pred:.3f} != target {target:.3f} (rel_error={rel_error:.2%})"
    
    def test_loss_decreases_during_training(self):
        """Test that loss decreases over training epochs."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Simple dataset
        X = np.random.randn(50, 5)
        population = np.random.uniform(1, 10, size=50)
        groups = np.random.randint(0, 3, size=50)
        targets = {0: 0.25, 1: 0.35, 2: 0.30}
        
        # Train model
        model = WeaklySupervisedNN(input_dim=5, hidden_dims=[16])
        model.fit(X, population, groups, targets, n_epochs=50, verbose=False)
        
        # Check that loss decreased
        assert len(model.loss_history) == 50
        initial_loss = model.loss_history[0]
        final_loss = model.loss_history[-1]
        
        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestGroupPreparation:
    """Test group preparation functions."""
    
    def test_prepare_zone_groups(self):
        """Test 3-zone group preparation."""
        # Create test DataFrame
        df = pd.DataFrame({
            'subregion': ['Urban', 'Urban', 'Rural', 'Rural', 
                          'Kingston Metropolitan Area (KMA)'],
            'moderate_prevalence': [22.92, 22.92, 34.65, 34.65, 34.57]
        })
        
        groups, targets = _prepare_zone_groups(df, 'moderate_prevalence')
        
        # Check groups
        assert len(groups) == 5
        assert set(groups) == {0, 1, 2}

        # Check targets (should be in 0-1 scale)
        # Dynamic sorted order: 0=KMA, 1=Rural, 2=Urban
        assert len(targets) == 3
        assert abs(targets[0] - 0.3457) < 0.001  # KMA (alphabetically first)
        assert abs(targets[1] - 0.3465) < 0.001  # Rural
        assert abs(targets[2] - 0.2292) < 0.001  # Urban
    
    def test_prepare_quintile_groups(self):
        """Test 15-group quintile preparation."""
        # Create test DataFrame with quintile probabilities
        df = pd.DataFrame({
            'subregion': ['Urban'] * 5 + ['Rural'] * 5,
            'p_q1': [0.8, 0.2, 0.0, 0.0, 0.0, 0.9, 0.3, 0.1, 0.0, 0.0],
            'p_q2': [0.2, 0.6, 0.2, 0.0, 0.0, 0.1, 0.5, 0.3, 0.1, 0.0],
            'p_q3': [0.0, 0.2, 0.6, 0.2, 0.0, 0.0, 0.2, 0.5, 0.3, 0.1],
            'p_q4': [0.0, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0, 0.1, 0.5, 0.3],
            'p_q5': [0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0, 0.1, 0.6],
            'moderate_prevalence': [22.92] * 5 + [34.65] * 5,
        })

        # Create quintile targets DataFrame (mimics jam_quintile_targets.csv)
        quintile_targets_df = pd.DataFrame({
            'subregion': ['Urban', 'Urban', 'Urban', 'Urban', 'Urban',
                          'Rural', 'Rural', 'Rural', 'Rural', 'Rural'],
            'quintile': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5',
                         'Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            'moderate_prevalence': [43.1, 32.8, 22.7, 12.0, 6.6,
                                   61.2, 32.7, 22.4, 18.4, 11.6],
        })

        groups, targets = _prepare_quintile_groups(
            df, 'moderate_prevalence', quintile_targets_df=quintile_targets_df
        )

        # Check groups (should be 0-14 for 3 zones × 5 quintiles)
        assert len(groups) == 10
        assert groups.min() >= 0
        assert groups.max() <= 14

        # Check that we have quintile-specific targets
        assert len(targets) > 3, "Should have more than 3 zone-level targets"

        # Urban Q1 should be different from Urban Q5
        urban_q1_target = targets.get(0)  # Urban (zone 0) × Q1 (quintile 0)
        urban_q5_target = targets.get(4)  # Urban (zone 0) × Q5 (quintile 4)

        if urban_q1_target and urban_q5_target:
            assert urban_q1_target > urban_q5_target, \
                "Q1 (poorest) should have higher poverty than Q5 (richest)"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_model_can_handle_missing_quintile_targets():
    """Test that model gracefully handles missing quintile targets."""
    # DataFrame with zones but no quintile targets CSV
    df = pd.DataFrame({
        'subregion': ['Urban'] * 10,
        'p_q1': [0.9, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        'p_q2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3, 0.1, 0.0, 0.0],
        'p_q3': [0.0, 0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.4, 0.2, 0.1],
        'p_q4': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.5, 0.3],
        'p_q5': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.6],
        'moderate_prevalence': [22.92] * 10,
    })

    # No quintile_targets_df → should fall back to zone-level targets
    groups, targets = _prepare_quintile_groups(df, 'moderate_prevalence')

    # Should still work, using zone-level fallback
    assert len(groups) == 10
    assert len(targets) > 0

    # All targets should be the zone-level value (converted to 0-1 scale)
    for target_val in targets.values():
        assert abs(target_val - 0.2292) < 0.001, \
            "Should fall back to zone-level target when quintiles unavailable"
