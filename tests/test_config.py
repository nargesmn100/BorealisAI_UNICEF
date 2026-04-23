"""Tests for config_loader.py."""

import os
import pytest

from src.utils.config_loader import load_config


class TestConfigLoader:
    def test_config_loads(self):
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "project_root" in cfg

    def test_required_sections(self):
        cfg = load_config()
        for section in ["paths", "geo", "targets", "modeling", "evaluation", "logging"]:
            assert section in cfg, f"Missing required section: {section}"

    def test_required_path_keys(self):
        cfg = load_config()
        required_keys = [
            "raw_data_dir", "rwi_csv", "interim_dir", "outputs_dir",
            "tables_dir", "maps_dir", "eval_dir",
        ]
        for key in required_keys:
            assert key in cfg["paths"], f"Missing path key: {key}"

    def test_paths_are_absolute(self):
        cfg = load_config()
        for key, val in cfg["paths"].items():
            if isinstance(val, str):
                assert os.path.isabs(val), f"Path '{key}' is not absolute: {val}"

    def test_feature_list(self):
        cfg = load_config()
        features = cfg["modeling"]["features"]
        assert len(features) >= 5
        assert "rwi" in features

    def test_bootstrap_range(self):
        cfg = load_config()
        n_boot = cfg["modeling"]["uncertainty"]["n_bootstrap"]
        assert n_boot >= 2, f"n_bootstrap={n_boot} is too low"
