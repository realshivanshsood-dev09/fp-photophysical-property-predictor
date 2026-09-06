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

DEFAULT_V3_CONFIG: dict[str, Any] = {
    "experiment": "v3_frozen_esm2_ridge",
    "scope": "all_families",
    "state_policy": "default_only",
    "target": {"name": "brightness", "task": "regression", "transformation": "log10_brightness_plus_one"},
    "features": {
        "name": "frozen_esm2",
        "model_name": "facebook/esm2_t12_35M_UR50D",
        "revision": "6fbf070e65b0b7291e7bbcd451118c216cff79d8",
        "pooling": "final_hidden_state_residue_mean",
        "embedding_dimension": 480,
        "batch_size": 4,
        "device": "auto",
    },
    "estimator": {"name": "ridge", "alpha": 1.0, "scaling": "standard_scaler_train_fold_only"},
    "split": {"strategy": "frozen_v2b_representative_cluster_kfold", "folds": 5, "seed": 42},
    "frozen_v2b": {
        "artifact_dir": "results/v2b_representative_regression/run_20260906T155344Z",
        "source_data_sha256": "a0b3490e8e14d5da81c6ccb5b4750aedbcfc04dee48fb9a23c0904aeea8f7568",
        "representative_assignment_sha256": "729b4818111719828094e41ca050781a24f98f40998e746ef52f48a224af6064",
        "fold_assignments_sha256": "fb4c39c0b8e05a32c8e1ad50b5d7fef74d50448734cb3ffdf865318abb00716a",
        "cluster_assignments_sha256": "bcd50dd68f39c259613216c2b3810e1b361af6a6bb3f41974592b51499a16414",
        "metadata_sha256": "70d5bf2448845d3fb09645c5dcbac6de571cdbc40e4c576a075632d48d99f257",
    },
    "models": ["mean_baseline", "median_baseline", "composition_ridge", "esm2_ridge"],
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


def validate_v3_config(config: dict[str, Any]) -> dict[str, Any]:
    if config["experiment"] != "v3_frozen_esm2_ridge":
        raise ValueError("V3 requires experiment 'v3_frozen_esm2_ridge'.")
    if config["scope"] != "all_families" or config["state_policy"] != "default_only":
        raise ValueError("V3 uses the frozen all_families/default_only V2b dataset only.")
    if config["target"] != {"name": "brightness", "task": "regression", "transformation": "log10_brightness_plus_one"}:
        raise ValueError("V3 supports only frozen log10(brightness + 1) regression.")
    features = config["features"]
    if set(features) != {"name", "model_name", "revision", "pooling", "embedding_dimension", "batch_size", "device"}:
        raise ValueError("V3 embedding configuration contains unsupported settings.")
    expected_features = {
        "name": "frozen_esm2", "model_name": "facebook/esm2_t12_35M_UR50D",
        "revision": "6fbf070e65b0b7291e7bbcd451118c216cff79d8",
        "pooling": "final_hidden_state_residue_mean", "embedding_dimension": 480,
    }
    if any(features.get(key) != value for key, value in expected_features.items()):
        raise ValueError("V3 uses one pinned ESM-2 final-layer residue-mean representation only.")
    if int(features.get("batch_size", 0)) < 1 or features.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("V3 embedding batch_size/device configuration is invalid.")
    if config["estimator"] != {"name": "ridge", "alpha": 1.0, "scaling": "standard_scaler_train_fold_only"}:
        raise ValueError("V3 uses fixed StandardScaler plus Ridge(alpha=1.0) only.")
    if config["split"] != {"strategy": "frozen_v2b_representative_cluster_kfold", "folds": 5, "seed": 42}:
        raise ValueError("V3 uses the exact frozen V2b five-fold assignments only.")
    if config["models"] != DEFAULT_V3_CONFIG["models"]:
        raise ValueError("V3 model set is fixed before embedding inference.")
    required_hashes = {
        "source_data_sha256", "representative_assignment_sha256", "fold_assignments_sha256",
        "cluster_assignments_sha256", "metadata_sha256",
    }
    frozen = config["frozen_v2b"]
    if not isinstance(frozen.get("artifact_dir"), str) or required_hashes.difference(frozen):
        raise ValueError("V3 requires complete frozen V2b artifact provenance.")
    for key in required_hashes:
        value = frozen[key]
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError(f"V3 frozen provenance {key!r} must be a SHA-256 hex digest.")
    return config


def load_v3_config(path: str | Path = "configs/v3_frozen_esm2_ridge.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    if not isinstance(supplied, dict):
        raise ValueError("Configuration root must be a mapping.")
    return validate_v3_config(_merge(DEFAULT_V3_CONFIG, supplied))
