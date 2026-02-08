"""Configuration loading and saving for APAB."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from apab.core.schemas import ProjectConfig

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate an apab.yaml config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config file: expected a YAML mapping, got {type(raw).__name__}")

    config = ProjectConfig.model_validate(raw)

    # Validate provider availability at load time
    from apab.agent.provider_registry import validate_provider

    if not validate_provider(config.llm.provider):
        logger.warning(
            "LLM provider %r is not available — agent commands will fail",
            config.llm.provider,
        )

    return config


def save_config(config: ProjectConfig, path: str | Path) -> None:
    """Save a ProjectConfig to a YAML file."""
    path = Path(path)
    data = config.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
