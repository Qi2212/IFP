"""
YAML Configuration Loader
==========================
Loads and merges YAML configuration files for extract, train, and inference.
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file (absolute or relative to project root).

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty or invalid YAML config: {config_path}")

    return config


def merge_configs(base: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge an override config into a base config (shallow merge for top-level keys).

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary (optional).

    Returns:
        dict: Merged configuration.
    """
    if override is None:
        return base

    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value

    return merged


def resolve_relative_path(config_path: str, relative_path: str, project_root: str = None) -> str:
    """
    Resolve a path relative to the config file location.

    Args:
        config_path: Path to the config file.
        relative_path: Path that may be relative.
        project_root: Optional explicit project root override.

    Returns:
        str: Resolved absolute or project-relative path.
    """
    if os.path.isabs(relative_path):
        return relative_path

    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(config_path))

    return os.path.normpath(os.path.join(project_root, relative_path))
