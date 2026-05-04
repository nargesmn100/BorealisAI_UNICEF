"""Tests for config_loader.py."""

import os
import pytest

from src.utils.config_loader import load_config, find_project_root


def _load_any_config():
    """Load the first available config: default, then NGA, then any in config/."""
    root = find_project_root()
    candidates = [
        os.path.join(root, "config", "config.yaml"),
        os.path.join(root, "config", "config_nga.yaml"),
    ]
    # also pick up any other config_*.yaml present
    config_dir = os.path.join(root, "config")
    if os.path.isdir(config_dir):
        for fname in sorted(os.listdir(config_dir)):
            if fname.startswith("config") and fname.endswith(".yaml"):
                candidates.append(os.path.join(config_dir, fname))

    for path in candidates:
        if os.path.isfile(path):
            return load_config(path)
    pytest.skip("No config file found — skipping config tests.")


class TestConfigLoader:
    def test_config_loads(self):
        cfg = _load_any_config()
        assert isinstance(cfg, dict)
        assert "project_root" in cfg

    def test_required_sections(self):
        cfg = _load_any_config()
        for section in ["paths", "geo", "targets", "modeling", "evaluation", "logging"]:
            assert section in cfg, f"Missing required section: {section}"

    def test_required_path_keys(self):
        cfg = _load_any_config()
        required_keys = [
            "raw_data_dir", "interim_dir", "outputs_dir",
            "tables_dir", "maps_dir", "eval_dir",
        ]
        for key in required_keys:
            assert key in cfg["paths"], f"Missing path key: {key}"

    def test_paths_are_absolute(self):
        cfg = _load_any_config()
        for key, val in cfg["paths"].items():
            if isinstance(val, str):
                assert os.path.isabs(val), f"Path '{key}' is not absolute: {val}"

    def test_feature_list(self):
        cfg = _load_any_config()
        features = cfg["modeling"]["features"]
        assert len(features) >= 5
        assert "rwi" in features

    def test_bootstrap_range(self):
        cfg = _load_any_config()
        n_boot = cfg["modeling"]["uncertainty"]["n_bootstrap"]
        assert n_boot >= 2, f"n_bootstrap={n_boot} is too low"
