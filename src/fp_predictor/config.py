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
    "features": {"name": "composition", "version": "1.1"},
    "split": {"strategy": "sequence_neighborhood_group_kfold", "folds": 5, "seed": 42},
    "models": [
        "majority",
        "stratified_random",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ],
}

DEFAULT_V2_CONFIG: dict[str, Any] = {
    "experiment": "v2_homology_regression",
    "scope": "all_families",
    "state_policy": "default_only",
    "target": {"name": "brightness", "task": "regression", "transformation": "log10_brightness_plus_one"},
    "features": {"name": "composition", "version": "1.1"},
    "split": {"strategy": "homology_cluster_kfold", "folds": 5, "seed": 42},
    "homology": {
        "alignment_method": "needleman_wunsch_global_linear_gap",
        "identity_definition": "identical_residue_columns / all_global_alignment_columns_including_gaps",
        "identity_threshold": 0.70,
    },
    "models": [
        "mean_baseline", "median_baseline", "ridge", "elastic_net",
        "random_forest_regressor", "gradient_boosting_regressor",
    ],
}

DEFAULT_V2B_CONFIG: dict[str, Any] = {
    **deepcopy(DEFAULT_V2_CONFIG),
    "experiment": "v2b_representative_regression",
    "split": {"strategy": "representative_cluster_kfold", "folds": 5, "seed": 42},
    "homology": {
        "clustering_strategy": "deterministic_representative_sequence_similarity",
        "alignment_method": "needleman_wunsch_global_linear_gap",
        "identity_definition": "identical_residue_columns / all_global_alignment_columns_including_gaps",
        "identity_threshold": 0.70,
    },
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


def validate_v2_config(config: dict[str, Any]) -> dict[str, Any]:
    if config["experiment"] != "v2_homology_regression":
        raise ValueError("V2 requires experiment 'v2_homology_regression'.")
    if config["scope"] != "all_families" or config["state_policy"] != "default_only":
        raise ValueError("The prespecified V2 dataset is all_families with default_only states.")
    if config["target"] != {"name": "brightness", "task": "regression", "transformation": "log10_brightness_plus_one"}:
        raise ValueError("V2 supports only log10(brightness + 1) regression.")
    if config["features"] != {"name": "composition", "version": "1.1"}:
        raise ValueError("V2 uses the fixed corrected composition feature set v1.1.")
    split = config["split"]
    if split["strategy"] != "homology_cluster_kfold" or int(split["folds"]) != 5 or int(split["seed"]) != 42:
        raise ValueError("V2 is prespecified as five-fold homology_cluster_kfold with seed 42.")
    homology = config["homology"]
    if homology["alignment_method"] != "needleman_wunsch_global_linear_gap":
        raise ValueError("V2 uses fixed global Needleman-Wunsch alignment.")
    if homology["identity_definition"] != "identical_residue_columns / all_global_alignment_columns_including_gaps":
        raise ValueError("V2 has one fixed alignment-identity definition.")
    if float(homology["identity_threshold"]) != 0.70:
        raise ValueError("V2 prespecifies an alignment-identity threshold of 0.70.")
    expected_models = DEFAULT_V2_CONFIG["models"]
    if config["models"] != expected_models:
        raise ValueError("V2 model set is fixed before cross-validation.")
    return config


def load_v2_config(path: str | Path = "configs/v2_homology_regression.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    if not isinstance(supplied, dict):
        raise ValueError("Configuration root must be a mapping.")
    return validate_v2_config(_merge(DEFAULT_V2_CONFIG, supplied))


def validate_v2b_config(config: dict[str, Any]) -> dict[str, Any]:
    if config["experiment"] != "v2b_representative_regression":
        raise ValueError("V2b requires experiment 'v2b_representative_regression'.")
    if config["scope"] != "all_families" or config["state_policy"] != "default_only":
        raise ValueError("The prespecified V2b dataset is all_families with default_only states.")
    if config["target"] != {"name": "brightness", "task": "regression", "transformation": "log10_brightness_plus_one"}:
        raise ValueError("V2b supports only log10(brightness + 1) regression.")
    if config["features"] != {"name": "composition", "version": "1.1"}:
        raise ValueError("V2b uses the fixed corrected composition feature set v1.1.")
    split = config["split"]
    if split["strategy"] != "representative_cluster_kfold" or int(split["folds"]) != 5 or int(split["seed"]) != 42:
        raise ValueError("V2b is prespecified as five-fold representative_cluster_kfold with seed 42.")
    homology = config["homology"]
    if homology.get("clustering_strategy") != "deterministic_representative_sequence_similarity":
        raise ValueError("V2b uses only deterministic representative-based clustering.")
    if homology["alignment_method"] != "needleman_wunsch_global_linear_gap":
        raise ValueError("V2b uses fixed global Needleman-Wunsch alignment.")
    if homology["identity_definition"] != "identical_residue_columns / all_global_alignment_columns_including_gaps":
        raise ValueError("V2b has one fixed alignment-identity definition.")
    if float(homology["identity_threshold"]) != 0.70:
        raise ValueError("V2b prespecifies an alignment-identity threshold of 0.70.")
    if config["models"] != DEFAULT_V2_CONFIG["models"]:
        raise ValueError("V2b model set is fixed before cross-validation.")
    return config


def load_v2b_config(path: str | Path = "configs/v2b_representative_regression.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    if not isinstance(supplied, dict):
        raise ValueError("Configuration root must be a mapping.")
    return validate_v2b_config(_merge(DEFAULT_V2B_CONFIG, supplied))
