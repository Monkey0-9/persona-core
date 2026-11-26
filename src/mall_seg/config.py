"""Pydantic-validated configuration loader for the Mall Segmentation Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mall_seg.contracts import (
    ApiConfig,
    AppConfig,
    ArtifactsConfig,
    ClusteringConfig,
    DataConfig,
    DriftConfig,
    ExplainabilityConfig,
    FeaturesConfig,
    PrivacyConfig,
    StabilityConfig,
    load_config,
)

DEFAULT_CONFIG_PATH = Path("configs/pipeline_config.yaml")


def get_config(config_path: str | Path | None = None) -> AppConfig:
    """Load and validate pipeline configuration from YAML."""
    return load_config(config_path)


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    """Serialize AppConfig into a plain JSON/YAML-safe dictionary."""
    return cfg.model_dump(mode="python")


def save_config_yaml(cfg: AppConfig, output_path: str | Path) -> Path:
    """Persist a validated AppConfig cleanly to YAML."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
    return target


__all__ = [
    "ApiConfig",
    "AppConfig",
    "ArtifactsConfig",
    "ClusteringConfig",
    "DEFAULT_CONFIG_PATH",
    "DataConfig",
    "DriftConfig",
    "ExplainabilityConfig",
    "FeaturesConfig",
    "PrivacyConfig",
    "StabilityConfig",
    "config_to_dict",
    "get_config",
    "load_config",
    "save_config_yaml",
]
