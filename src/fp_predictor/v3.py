"""Frozen-fold V3 ESM-2 brightness regression experiment.

V3 intentionally reuses the already-computed V2b partition.  It neither
re-clusters sequences nor chooses models from outer-fold performance.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import CompositionFeaturizer, ESM2Featurizer, EmbeddingSpec
from .regression import (
    V2_ARTIFACT_VERSION,
    _LocationBaseline,
    aggregate_regression_metrics,
    make_regressor,
    model_parameters,
    regression_metrics,
    transform_brightness,
)

V3_ARTIFACT_VERSION = "1"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_frozen_v2b_context(source: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Load V2b partitions verbatim and prove they match the V3 input."""
    frozen = config["frozen_v2b"]
    artifact_dir = Path(frozen["artifact_dir"])
    paths = {
        "metadata": artifact_dir / "metadata.json",
        "fold_assignments": artifact_dir / "fold_assignments.csv",
        "cluster_assignments": artifact_dir / "cluster_assignments.csv",
        "metrics": artifact_dir / "metrics.json",
    }
    absent = [name for name, path in paths.items() if not path.is_file()]
    if absent:
        raise ValueError(f"V3 frozen V2b artifact is incomplete: missing {sorted(absent)}.")
    observed_hashes = {
        "source_data_sha256": _hash_file(source),
        "metadata_sha256": _hash_file(paths["metadata"]),
        "fold_assignments_sha256": _hash_file(paths["fold_assignments"]),
        "cluster_assignments_sha256": _hash_file(paths["cluster_assignments"]),
    }
    for name, observed in observed_hashes.items():
        if observed.lower() != frozen[name].lower():
            raise ValueError(f"V3 refuses non-frozen input: {name} does not match the configured V2b digest.")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("experiment") != "v2b_representative_regression" or metadata.get("artifact_version") != V2_ARTIFACT_VERSION:
        raise ValueError("V3 frozen reference is not a compatible V2b artifact.")
    if metadata.get("source_data_sha256", "").lower() != observed_hashes["source_data_sha256"].lower():
        raise ValueError("V3 input data do not match the dataset recorded by V2b.")
    if metadata.get("clustering", {}).get("assignment_sha256", "").lower() != frozen["representative_assignment_sha256"].lower():
        raise ValueError("V3 reference does not have the configured V2b representative assignment hash.")
    if metadata.get("cleaned_records") != 601 or metadata.get("scope") != "all_families" or metadata.get("state_policy") != "default_only":
        raise ValueError("V3 accepts only the frozen 601-observation all-family/default-state V2b cohort.")

    frame = pd.read_csv(source)
    required = {"record_id", "sequence", "brightness"}
    missing = required.difference(frame.columns)
    if missing or len(frame) != 601 or frame["record_id"].astype(str).duplicated().any():
        raise ValueError("V3 input must be the unique-record 601-observation frozen V2b dataset.")
    frame = frame.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    assignments = pd.read_csv(paths["fold_assignments"])
    cluster_assignments = pd.read_csv(paths["cluster_assignments"])
    assignment_columns = {"fold", "record_id", "representative_cluster_id", "representative_record_id"}
    cluster_columns = assignment_columns | {"representative_identity"}
    if set(assignments.columns) != assignment_columns or set(cluster_assignments.columns) != cluster_columns:
        raise ValueError("V3 frozen assignment files have an incompatible schema.")
    assignments["record_id"] = assignments["record_id"].astype(str)
    cluster_assignments["record_id"] = cluster_assignments["record_id"].astype(str)
    if len(assignments) != 601 or assignments["record_id"].duplicated().any() or set(assignments["record_id"]) != set(frame["record_id"]):
        raise ValueError("V3 input record IDs do not exactly match frozen V2b fold assignments.")
    if len(cluster_assignments) != 601 or cluster_assignments["record_id"].duplicated().any():
        raise ValueError("V3 frozen V2b cluster assignments are incomplete.")
    comparable = cluster_assignments[list(assignment_columns)].sort_values("record_id").reset_index(drop=True)
    expected = assignments[list(assignment_columns)].sort_values("record_id").reset_index(drop=True)
    if not comparable.equals(expected):
        raise ValueError("V3 fold and cluster assignments disagree inside the frozen V2b artifact.")
    if sorted(assignments["fold"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("V3 frozen V2b reference must contain exactly five folds.")
    cluster_fold_count = assignments.groupby("representative_cluster_id")["fold"].nunique()
    if int(cluster_fold_count.max()) != 1:
        raise ValueError("V3 refuses frozen assignments with a representative-cluster fold overlap.")
    assigned = frame.merge(
        assignments[["record_id", "fold", "representative_cluster_id", "representative_record_id"]],
        on="record_id", how="left", validate="one_to_one", sort=False,
    )
    if assigned["fold"].isna().any():
        raise ValueError("V3 could not align every frozen V2b record to a fold.")
    return assigned, assignments, metadata, paths


def _embedding_ridge() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=1.0))])


def _make_model(name: str):
    if name == "mean_baseline":
        return _LocationBaseline("mean")
    if name == "median_baseline":
        return _LocationBaseline("median")
    if name == "composition_ridge":
        return make_regressor("ridge", seed=42)
    if name == "esm2_ridge":
        return _embedding_ridge()
    raise ValueError(f"Unsupported V3 model {name!r}.")


def load_v3_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = {"artifact_version", "model", "embedding_spec", "metadata", "target_transformation"}
    if not isinstance(artifact, dict) or required.difference(artifact):
        raise ValueError("V3 artifact is incomplete or incompatible.")
    if artifact["artifact_version"] != V3_ARTIFACT_VERSION:
        raise ValueError("V3 artifact version is incompatible.")
    if artifact["target_transformation"] != "log10(brightness + 1)":
        raise ValueError("V3 artifact has an incompatible target transformation.")
    return artifact


def train_v3_experiment(
    data_path: str | Path,
    config: dict[str, Any],
    results_dir: str | Path = "results/v3_frozen_esm2_ridge",
    cache_dir: str | Path = "data/cache/esm2_v3",
    source_metadata: dict[str, Any] | None = None,
    cleaning_summary: dict[str, Any] | None = None,
    *,
    embedder: ESM2Featurizer | None = None,
) -> Path:
    """Run the one pre-specified, frozen-fold V3 primary experiment."""
    source = Path(data_path)
    assigned, assignments, v2b_metadata, paths = _load_frozen_v2b_context(source, config)
    sequences = assigned["sequence"].astype(str).to_numpy()
    transformed = transform_brightness(assigned["brightness"].astype(float).to_numpy())
    feature_config = config["features"]
    spec = EmbeddingSpec(
        model_name=feature_config["model_name"], revision=feature_config["revision"],
        pooling=feature_config["pooling"], embedding_dimension=int(feature_config["embedding_dimension"]),
        batch_size=int(feature_config["batch_size"]), device=feature_config["device"],
    )
    embedder = embedder or ESM2Featurizer(spec, cache_dir)
    embeddings = np.asarray(embedder.transform(sequences), dtype=np.float32)
    if embeddings.shape != (len(assigned), 480) or not np.isfinite(embeddings).all():
        raise ValueError("V3 embedder did not return one finite 480-dimensional vector per frozen observation.")

    model_results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for name in config["models"]:
        folds: list[dict[str, Any]] = []
        for fold_number in range(1, 6):
            validation_mask = assigned["fold"].to_numpy(dtype=int) == fold_number
            validation_index = np.flatnonzero(validation_mask)
            train_index = np.flatnonzero(~validation_mask)
            model = _make_model(name)
            if name == "composition_ridge":
                train_x, validation_x = sequences[train_index], sequences[validation_index]
            elif name == "esm2_ridge":
                train_x, validation_x = embeddings[train_index], embeddings[validation_index]
            else:
                train_x, validation_x = embeddings[train_index], embeddings[validation_index]
            model.fit(train_x, transformed[train_index])
            prediction = np.asarray(model.predict(validation_x), dtype=float)
            validation_clusters = sorted(set(assigned.iloc[validation_index]["representative_cluster_id"]))
            train_clusters = sorted(set(assigned.iloc[train_index]["representative_cluster_id"]))
            overlap = set(train_clusters).intersection(validation_clusters)
            if overlap:
                raise RuntimeError(f"V3 frozen V2b cluster overlap in fold {fold_number}: {sorted(overlap)}")
            folds.append({
                "fold": fold_number, "train_size": int(len(train_index)), "validation_size": int(len(validation_index)),
                "train_clusters": train_clusters, "validation_clusters": validation_clusters,
                "metrics": regression_metrics(transformed[validation_index], prediction),
            })
            for index, value in zip(validation_index, prediction, strict=True):
                prediction_rows.append({
                    "model": name, "fold": fold_number, "record_id": assigned.iloc[index]["record_id"],
                    "representative_cluster_id": assigned.iloc[index]["representative_cluster_id"],
                    "y_log10_brightness_plus_one": float(transformed[index]), "prediction_log10_brightness_plus_one": float(value),
                })
        model_results[name] = {
            "parameters": model_parameters(_make_model(name)), "folds": folds,
            "aggregate": aggregate_regression_metrics(folds),
        }

    historical_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    historical_gradient_boosting = historical_metrics.get("gradient_boosting_regressor")
    if historical_gradient_boosting is None:
        raise ValueError("Frozen V2b metrics do not contain the required Gradient Boosting historical benchmark.")
    final_embedding_model = _embedding_ridge().fit(embeddings, transformed)
    final_composition_model = make_regressor("ridge", seed=42).fit(sequences, transformed)
    embedding_provenance = embedder.provenance(sequences)
    run_directory = Path(results_dir) / _run_id()
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(paths["fold_assignments"], run_directory / "frozen_v2b_fold_assignments.csv")
    shutil.copyfile(paths["cluster_assignments"], run_directory / "frozen_v2b_cluster_assignments.csv")
    (run_directory / "embedding_manifest.json").write_text(json.dumps(embedding_provenance, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(run_directory / "predictions.csv", index=False)
    frozen_provenance = {
        "artifact_dir": str(Path(config["frozen_v2b"]["artifact_dir"])),
        "source_data_sha256": _hash_file(source),
        "metadata_sha256": _hash_file(paths["metadata"]),
        "fold_assignments_sha256": _hash_file(paths["fold_assignments"]),
        "cluster_assignments_sha256": _hash_file(paths["cluster_assignments"]),
        "representative_assignment_sha256": v2b_metadata["clustering"]["assignment_sha256"],
        "verified_record_count": int(len(assignments)),
        "verified_fold_sizes": {
            str(fold): int((assignments["fold"] == fold).sum()) for fold in range(1, 6)
        },
        "verification": "input data, reference metadata, cluster assignments, and fold assignments matched configured SHA-256 values",
    }
    metadata = {
        "experiment": "v3_frozen_esm2_ridge", "artifact_version": V3_ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_data": str(source), "source_data_sha256": _hash_file(source),
        "source": source_metadata or {}, "cleaning_summary": cleaning_summary or {},
        "raw_records": int((source_metadata or {}).get("record_count", len(assigned))), "cleaned_records": int(len(assigned)),
        "scope": config["scope"], "state_policy": config["state_policy"], "target": config["target"],
        "target_transformation": "log10(brightness + 1)", "features": embedding_provenance,
        "matched_composition_baseline": {
            "name": "composition_v1.1_ridge", "feature_version": CompositionFeaturizer.feature_version,
            "feature_names": CompositionFeaturizer().get_feature_names_out().tolist(), "alpha": 1.0,
            "scaling": "standard_scaler_train_fold_only",
        },
        "split": config["split"], "frozen_v2b": frozen_provenance,
        "model_selection": {"performed": False, "reason": "one V3 representation and one fixed Ridge estimator were prespecified"},
        "historical_v2b_gradient_boosting": {
            "artifact_dir": str(Path(config["frozen_v2b"]["artifact_dir"])),
            "model": "gradient_boosting_regressor", "metrics": historical_gradient_boosting,
        },
        "python": sys.version, "platform": platform.platform(),
        "package_versions": {"joblib": joblib.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "protocol": {
            "encoder_training": "frozen inference-only; no fine-tuning or adapters",
            "dimensionality_reduction": "none", "feature_selection": "none", "embedding_tree_models": "none",
            "sequence_truncation": "forbidden; sequence length is validated against the model limit",
        },
    }
    artifact = {
        "artifact_version": V3_ARTIFACT_VERSION, "model": final_embedding_model,
        "composition_baseline_model": final_composition_model, "embedding_spec": feature_config,
        "target_transformation": "log10(brightness + 1)", "metadata": metadata,
    }
    joblib.dump(artifact, run_directory / "model.joblib")
    (run_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_directory / "metrics.json").write_text(json.dumps({
        "models": model_results, "historical_v2b_gradient_boosting": historical_gradient_boosting,
    }, indent=2), encoding="utf-8")
    load_v3_artifact(run_directory / "model.joblib")
    return run_directory
