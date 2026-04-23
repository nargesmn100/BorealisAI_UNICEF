"""Tests for evaluation/metrics.py — all use synthetic data."""

import numpy as np
import pytest

from src.evaluation.metrics import mae, rmse, spearman_r, pearson_r, top_k_overlap


class TestMAE:
    def test_identical_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == 0.0

    def test_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 3.0])
        assert mae(a, b) == 2.0

    def test_nan_handling(self):
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert mae(a, b) == 0.0

    def test_all_nan(self):
        a = np.array([np.nan, np.nan])
        b = np.array([1.0, 2.0])
        assert np.isnan(mae(a, b))


class TestRMSE:
    def test_identical_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        assert rmse(a, a) == 0.0

    def test_known_value(self):
        a = np.array([0.0])
        b = np.array([3.0])
        assert rmse(a, b) == 3.0


class TestCorrelations:
    def test_perfect_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(pearson_r(a, a) - 1.0) < 1e-10
        assert abs(spearman_r(a, a) - 1.0) < 1e-10

    def test_negative_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(spearman_r(a, b) - (-1.0)) < 1e-10

    def test_insufficient_data(self):
        a = np.array([1.0, 2.0])
        assert np.isnan(pearson_r(a, a))


class TestTopKOverlap:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert top_k_overlap(a, a, k=3) == 1.0

    def test_disjoint(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        # top-2 of a = [4,5], top-2 of b = [0,1] — disjoint
        assert top_k_overlap(a, b, k=2) == 0.0

    def test_k_larger_than_array(self):
        a = np.array([1.0, 2.0])
        b = np.array([2.0, 1.0])
        result = top_k_overlap(a, b, k=5)
        assert 0.0 <= result <= 1.0
