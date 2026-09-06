"""V3 secondary Aequorea local-interpolation diagnostic experiment.

This diagnostic restricts evaluation strictly to the 154 observations of the
canonical Aequorea GFP cluster (cluster:WQUOO) identified in the frozen V2b
partition.

CRITICAL METHODOLOGICAL TERMINOLOGY:
This experiment is a within-cluster local interpolation diagnostic. It is NOT
lineage-aware prediction, NOT unseen-scaffold generalization, and NOT prospective
variant design.
"""

from __future__ import annotations

import hashlib
import json
import platform
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

from .clean import normalize_sequence
from .features import CompositionFeaturizer, ESM2Featurizer, EmbeddingSpec
from .regression import (
    _LocationBaseline,
    aggregate_regression_metrics,
    make_regressor,
    model_parameters,
    regression_metrics,
    transform_brightness,
)
from .split import grouped_folds
from .v3 import V3_ARTIFACT_VERSION, _hash_file, _load_frozen_v2b_context

DIAGNOSTIC_EXPERIMENT = "v3_aequorea_diagnostic"


def _run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sequence_sha256(sequence: str) -> str:
    norm = normalize_sequence(sequence)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def train_v3_aequorea_diagnostic(
    data_path: str | Path,
    config: dict[str, Any],
    results_dir: str | Path = "results/v3_aequorea_diagnostic",
    cache_dir: str | Path = "data/cache/esm2_v3",
    source_metadata: dict[str, Any] | None = None,
    cleaning_summary: dict[str, Any] | None = None,
    *,
    embedder: ESM2Featurizer | None = None,
) -> Path:
    """Run 5-fold local-interpolation CV on the 154 Aequorea cluster observations."""
    source = Path(data_path)
    assigned, assignments, v2b_metadata, paths = _load_frozen_v2b_context(source, config)
    target_cluster = config.get("target_cluster_id", "cluster:WQUOO")
    aequorea_mask = assigned["representative_cluster_id"] == target_cluster
    df_aeq = assigned[aequorea_mask].copy().reset_index(drop=True)

    if len(df_aeq) != 154:
        raise ValueError(f"Expected exactly 154 Aequorea cluster observations, found {len(df_aeq)}.")

    df_aeq["sequence_group_id"] = df_aeq["sequence"].apply(_sequence_sha256)
    unique_groups = df_aeq["sequence_group_id"].nunique()
    groups = df_aeq["sequence_group_id"].to_numpy()
    splits = list(grouped_folds(groups, int(config["split"]["folds"]), seed=int(config["split"]["seed"])))

    sequences = df_aeq["sequence"].astype(str).to_numpy()
    transformed = transform_brightness(df_aeq["brightness"].astype(float).to_numpy())

    feature_config = config["features"]
    spec = EmbeddingSpec(
        model_name=feature_config["model_name"],
        revision=feature_config["revision"],
        pooling=feature_config["pooling"],
        embedding_dimension=int(feature_config["embedding_dimension"]),
        batch_size=int(feature_config["batch_size"]),
        device=feature_config["device"],
    )
    embedder = embedder or ESM2Featurizer(spec, cache_dir)
    embeddings = np.asarray(embedder.transform(sequences), dtype=np.float32)

    model_results: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    fold_assignment_rows: list[dict[str, Any]] = []

    for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
        for idx in validation_index:
            fold_assignment_rows.append({
                "fold": fold_number,
                "record_id": df_aeq.iloc[idx]["record_id"],
                "sequence_sha256": groups[idx],
                "name": df_aeq.iloc[idx].get("name", ""),
            })

    for name in config["models"]:
        folds: list[dict[str, Any]] = []
        for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
            val_groups = set(groups[validation_index])
            train_groups = set(groups[train_index])
            overlap = val_groups.intersection(train_groups)
            if overlap:
                raise RuntimeError(f"Sequence group overlap detected in fold {fold_number}: {sorted(overlap)}")

            if name == "mean_baseline":
                model = _LocationBaseline("mean")
                model.fit(sequences[train_index], transformed[train_index])
                prediction = model.predict(sequences[validation_index])
            elif name == "median_baseline":
                model = _LocationBaseline("median")
                model.fit(sequences[train_index], transformed[train_index])
                prediction = model.predict(sequences[validation_index])
            elif name == "composition_ridge":
                model = Pipeline([
                    ("features", CompositionFeaturizer()),
                    ("scale", StandardScaler()),
                    ("regressor", Ridge(alpha=1.0, random_state=int(config["split"]["seed"]) + fold_number)),
                ])
                model.fit(sequences[train_index], transformed[train_index])
                prediction = model.predict(sequences[validation_index])
            elif name == "esm2_ridge":
                model = Pipeline([
                    ("scale", StandardScaler()),
                    ("regressor", Ridge(alpha=1.0, random_state=int(config["split"]["seed"]) + fold_number)),
                ])
                model.fit(embeddings[train_index], transformed[train_index])
                prediction = model.predict(embeddings[validation_index])
            else:
                raise ValueError(f"Unsupported diagnostic model {name!r}.")

            prediction = np.asarray(prediction, dtype=float)
            folds.append({
                "fold": fold_number,
                "train_size": int(len(train_index)),
                "validation_size": int(len(validation_index)),
                "train_sequence_groups": len(train_groups),
                "validation_sequence_groups": len(val_groups),
                "metrics": regression_metrics(transformed[validation_index], prediction),
            })

            for index, value in zip(validation_index, prediction, strict=True):
                prediction_rows.append({
                    "model": name,
                    "fold": fold_number,
                    "record_id": df_aeq.iloc[index]["record_id"],
                    "sequence_sha256": groups[index],
                    "y_log10_brightness_plus_one": float(transformed[index]),
                    "prediction_log10_brightness_plus_one": float(value),
                })

        model_results[name] = {
            "folds": folds,
            "aggregate": aggregate_regression_metrics(folds),
        }

    final_composition_model = Pipeline([
        ("features", CompositionFeaturizer()),
        ("scale", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=42)),
    ]).fit(sequences, transformed)

    final_esm2_model = Pipeline([
        ("scale", StandardScaler()),
        ("regressor", Ridge(alpha=1.0, random_state=42)),
    ]).fit(embeddings, transformed)

    embedding_provenance = embedder.provenance(sequences)
    run_directory = Path(results_dir) / _run_id()
    run_directory.mkdir(parents=True, exist_ok=False)

    pd.DataFrame(fold_assignment_rows).to_csv(run_directory / "fold_assignments.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run_directory / "predictions.csv", index=False)
    (run_directory / "embedding_manifest.json").write_text(json.dumps(embedding_provenance, indent=2), encoding="utf-8")

    metadata = {
        "experiment": DIAGNOSTIC_EXPERIMENT,
        "diagnostic_type": "aequorea_local_interpolation",
        "disclaimer": (
            "This is a local interpolation diagnostic within the Aequorea mutational neighborhood, "
            "NOT lineage-aware prediction, NOT unseen-scaffold generalization, and NOT prospective variant prediction."
        ),
        "artifact_version": V3_ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_data": str(source),
        "source_data_sha256": _hash_file(source),
        "scope": "aequorea_cluster_only",
        "target_cluster_id": target_cluster,
        "target_observations": len(df_aeq),
        "unique_sequence_groups": unique_groups,
        "target": config["target"],
        "target_transformation": "log10(brightness + 1)",
        "features": embedding_provenance,
        "matched_composition_baseline": {
            "name": "composition_v1.1_ridge",
            "feature_version": CompositionFeaturizer.feature_version,
            "alpha": 1.0,
            "scaling": "standard_scaler_train_fold_only",
        },
        "split": {
            "strategy": "exact_sequence_group_kfold",
            "folds": 5,
            "seed": 42,
            "grouping_criterion": "normalized_sequence_sha256",
        },
        "frozen_v2b_reference": {
            "artifact_dir": str(Path(config["frozen_v2b"]["artifact_dir"])),
            "source_data_sha256": _hash_file(source),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    artifact = {
        "artifact_version": V3_ARTIFACT_VERSION,
        "composition_model": final_composition_model,
        "esm2_model": final_esm2_model,
        "embedding_spec": feature_config,
        "target_transformation": "log10(brightness + 1)",
        "metadata": metadata,
    }
    joblib.dump(artifact, run_directory / "model.joblib")
    (run_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_directory / "metrics.json").write_text(json.dumps({"models": model_results}, indent=2), encoding="utf-8")
    return run_directory
