"""Versioned configuration loading and validation."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError
from .hashing import hash_object

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar/list values."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"Configuration does not exist: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration must be a mapping: {resolved}")
    return value


def load_config(source_config: str | Path) -> dict[str, Any]:
    """Load shared decisions, then the source mapping and explicit overrides."""
    source_path = Path(source_config).resolve()
    config: dict[str, Any] = {}
    for filename in ("cohort.yaml", "features.yaml", "models.yaml", "evaluation.yaml"):
        config = deep_merge(config, read_yaml(PROJECT_ROOT / "configs" / filename))
    source = read_yaml(source_path)
    overrides = source.pop("overrides", {})
    config = deep_merge(config, source)
    config = deep_merge(config, overrides)
    config["_meta"] = {
        "source_config": str(source_path),
        "project_root": str(PROJECT_ROOT),
    }
    validate_config(config)
    config["_meta"]["config_hash"] = hash_object(
        {key: value for key, value in config.items() if key != "_meta"}
    )
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def require_environment_reference(name: str) -> str:
    if not name or any(token in name.lower() for token in ("://", "password", "/")):
        raise ConfigurationError("Database configuration must name an environment variable")
    value = os.environ.get(name)
    if not value:
        raise ConfigurationError(f"Required environment variable is unset: {name}")
    return value


def validate_config(config: dict[str, Any]) -> None:
    required = ("dataset", "adapter", "source", "cohort", "features", "folds", "models", "evaluation")
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {missing}")
    if config["adapter"] not in {"chorus", "mimiciv"}:
        raise ConfigurationError(f"Unsupported adapter: {config['adapter']}")
    if not config["source"].get("mapping_confirmed", False):
        raise ConfigurationError("Source mapping must be explicitly confirmed")
    if config["folds"]["count"] != 5:
        raise ConfigurationError("The scientific design requires exactly five folds")
    if config["features"]["concept_count"] != 50:
        raise ConfigurationError("The scientific design requires 50 concepts per domain")
    if config["cohort"]["landmark_hours"] != 24:
        raise ConfigurationError("The scientific design requires a 24-hour landmark")
    if config["cohort"]["outcome_horizon_days"] != 30:
        raise ConfigurationError("The scientific design requires a 30-day outcome horizon")
    if config["evaluation"]["bootstrap"]["method"] != "percentile":
        raise ConfigurationError("Only the prespecified percentile bootstrap is allowed")
    matrix_names = list(config["matrices"])
    if len(matrix_names) != 8:
        raise ConfigurationError("Exactly eight feature matrices are required")
    model_names = config["models"]["frozen_order"]
    if model_names != [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "lightgbm",
    ]:
        raise ConfigurationError("Frozen model order has changed")
