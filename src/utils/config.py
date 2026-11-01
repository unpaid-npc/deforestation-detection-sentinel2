"""Configuration loading utilities.

All pipeline modules read their parameters through ``load_config`` so that paths,
dates and thresholds live in ``config/config.yaml`` rather than in the code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# src/utils/config.py -> parents[0]=utils, [1]=src, [2]=project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Optional path to a YAML file. Defaults to ``config/config.yaml``.

    Returns:
        The configuration as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain a YAML mapping.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    logger.debug("Loaded configuration from %s", config_path)
    return config