"""
Config loader utility.
Loads the YAML config file and resolves all paths relative to the project root.
"""

import logging
import os

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def find_project_root(marker_files=("config/config.yaml", "README.md", "src")):
    """
    Walk up from the current working directory to find the project root,
    identified by the presence of a known marker file or directory.

    Returns
    -------
    str
        Absolute path to the project root.

    Raises
    ------
    RuntimeError
        If no project root can be located.
    """
    cwd = os.path.abspath(os.getcwd())
    candidate = cwd
    for _ in range(10):  # max 10 levels up
        for marker in marker_files:
            if os.path.exists(os.path.join(candidate, marker)):
                return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    raise RuntimeError(
        f"Could not locate project root from '{cwd}'. "
        "Make sure you run the pipeline from within the project directory."
    )


def load_config(config_path: str | None = None) -> dict:
    """
    Load the YAML configuration file and resolve paths relative to project root.

    Parameters
    ----------
    config_path : str or None
        Path to the config.yaml file.  If None, the function looks for
        ``config/config.yaml`` relative to the detected project root.

    Returns
    -------
    dict
        Parsed configuration with an additional ``project_root`` key.
    """
    project_root = find_project_root()

    if config_path is None:
        config_path = os.path.join(project_root, "config", "config.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    cfg["project_root"] = project_root

    # Resolve all path values to absolute paths
    if "paths" in cfg:
        for key, rel_path in cfg["paths"].items():
            if isinstance(rel_path, str):
                cfg["paths"][key] = os.path.join(project_root, rel_path)

    validate_config(cfg)
    logger.debug("Config loaded from %s, project root: %s", config_path, project_root)
    return cfg


def validate_config(cfg: dict) -> None:
    """
    Validate that the config has required sections and keys.

    Raises ValueError for missing required sections/keys.
    Logs warnings for missing raw data paths (which may not exist on all machines).
    """
    required_sections = ["paths", "geo", "targets", "modeling", "evaluation", "logging"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Config missing required section: '{section}'")

    required_path_keys = [
        "raw_data_dir", "rwi_csv", "interim_dir", "outputs_dir",
        "tables_dir", "maps_dir", "eval_dir",
    ]
    for key in required_path_keys:
        if key not in cfg["paths"]:
            raise ValueError(f"Config paths missing required key: '{key}'")

    # Warn (not error) if raw data paths don't exist on disk
    raw_keys = ["rwi_csv", "population_raster", "gadm_gpkg"]
    for key in raw_keys:
        if key in cfg["paths"]:
            path = cfg["paths"][key]
            if isinstance(path, str) and not os.path.exists(path):
                logger.warning("Raw data path does not exist: %s = %s", key, path)

    # Validate numeric ranges
    n_bootstrap = cfg.get("modeling", {}).get("uncertainty", {}).get("n_bootstrap", 50)
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be >= 2, got {n_bootstrap}")


def get_available_features(cfg: dict, df: pd.DataFrame) -> list[str]:
    """
    Return the subset of configured features that actually exist in the DataFrame.

    Logs a warning for any configured features not present (e.g., nightlights
    raster wasn't on disk so the column was never created). Raises if fewer
    than 3 features survive.

    Parameters
    ----------
    cfg : dict
        Loaded config with cfg["modeling"]["features"] list.
    df : pd.DataFrame
        The modeling table or feature matrix.

    Returns
    -------
    list[str]
        Feature column names that are present in df.
    """
    configured = cfg["modeling"]["features"]
    available = [f for f in configured if f in df.columns]
    missing = [f for f in configured if f not in df.columns]
    if missing:
        logger.warning("Configured features not in DataFrame (skipping): %s", missing)
    if len(available) < 3:
        raise ValueError(
            f"Too few features available ({len(available)}/{len(configured)}). "
            f"Missing: {missing}. Check data pipeline."
        )
    return available


def setup_logging(cfg: dict) -> None:
    """
    Configure root logging based on config.

    Parameters
    ----------
    cfg : dict
        Loaded config dictionary.
    """
    log_cfg = cfg.get("logging", {})
    level_str = log_cfg.get("level", "INFO")
    fmt = log_cfg.get(
        "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format=fmt)
