"""Configuration loading and a deliberately small, validated schema."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "scope": "all_families",
    "state_policy": "default_only",
    "lineage_sidecar": None,
    "near_duplicate_identity": 0.95,
    "target": {
        "name": "brightness",
        "task": "classification",
        "n_classes": 3,
        "threshold_strategy": "fold_train_tertiles",
    },
    "features": {"name": "composition", "version": "1.0"},
    "split": {"strategy": "sequence_neighborhood_group_kfold", "folds": 5, "seed": 42},
    "models": [
        "majority",
        "stratified_random",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ],
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config["scope"] not in {"gfp_only", "all_families"}:
        raise ValueError("scope must be 'gfp_only' or 'all_families'.")
    if config["state_policy"] not in {"default_only", "all_states"}:
        raise ValueError("state_policy must be 'default_only' or 'all_states'.")
    threshold = config.get("near_duplicate_identity")
    if threshold is not None and not 0 < float(threshold) < 1:
        raise ValueError("near_duplicate_identity must be null or strictly between 0 and 1.")
    target = config["target"]
    if target["name"] != "brightness" or target["task"] != "classification":
        raise ValueError("v1 supports brightness classification only.")
    if target["n_classes"] != 3:
        raise ValueError("v1 supports exactly three operational brightness tiers.")
    if target["threshold_strategy"] != "fold_train_tertiles":
        raise ValueError("Only leakage-safe 'fold_train_tertiles' is supported in v1.")
    split = config["split"]
    if split["strategy"] != "sequence_neighborhood_group_kfold":
        raise ValueError("v1 supports 'sequence_neighborhood_group_kfold' only.")
    if int(split["folds"]) < 2:
        raise ValueError("split.folds must be at least 2.")
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, overlay it on defaults, and reject unsupported modes."""
    supplied: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            supplied = yaml.safe_load(handle) or {}
        if not isinstance(supplied, dict):
            raise ValueError("Configuration root must be a mapping.")
    return validate_config(_merge(DEFAULT_CONFIG, supplied))
