"""
Config loader utility.
Loads the YAML config file and resolves all paths relative to the project root.
"""

import os
import logging
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

    logger.debug("Config loaded from %s, project root: %s", config_path, project_root)
    return cfg


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
