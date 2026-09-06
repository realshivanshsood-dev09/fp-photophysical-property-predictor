"""Prespecified V2 continuous-brightness regression experiment."""

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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import CompositionFeaturizer
from .homology import (
    ALIGNMENT_METHOD,
    ALIGNMENT_SCORING,
    IDENTITY_DEFINITION,
    build_homology_clusters,
    build_representative_clusters,
)
from .split import grouped_folds

V2_ARTIFACT_VERSION = "1"
V2_MODEL_ORDER = {
    "mean_baseline": 0, "median_baseline": 1, "ridge": 2, "elastic_net": 3,
    "random_forest_regressor": 4, "gradient_boosting_regressor": 5,
}


class _LocationBaseline:
    def __init__(self, statistic: str) -> None:
        self.statistic = statistic

    def fit(self, X, y):
        values = np.asarray(y, dtype=float)
        self.value_ = float(values.mean() if self.statistic == "mean" else np.median(values))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_, dtype=float)


def transform_brightness(brightness) -> np.ndarray:
    """Apply the fixed V2 transform ``log10(brightness + 1)``."""
    values = np.asarray(brightness, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Brightness transformation requires finite non-negative values.")
    return np.log10(values + 1.0)


def inverse_transform_brightness(transformed) -> np.ndarray:
    """Invert V2's deterministic target transformation."""
    values = np.asarray(transformed, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Inverse brightness transformation requires finite values.")
    return np.power(10.0, values) - 1.0


def make_regressor(name: str, seed: int):
    """Return one of V2's fixed, prespecified estimators."""
    if name == "mean_baseline":
        return _LocationBaseline("mean")
    if name == "median_baseline":
        return _LocationBaseline("median")
    if name == "ridge":
        return Pipeline([("features", CompositionFeaturizer()), ("scale", StandardScaler()), ("regressor", Ridge(alpha=1.0))])
    if name == "elastic_net":
        return Pipeline([
            ("features", CompositionFeaturizer()), ("scale", StandardScaler()),
            ("regressor", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000, random_state=seed)),
        ])
    if name == "random_forest_regressor":
        return Pipeline([
            ("features", CompositionFeaturizer()),
            ("regressor", RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=seed, n_jobs=1,
            )),
        ])
    if name == "gradient_boosting_regressor":
        return Pipeline([
            ("features", CompositionFeaturizer()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=80, learning_rate=0.05, max_depth=2, min_samples_leaf=2, random_state=seed,
            )),
        ])
    raise ValueError(f"Unsupported V2 regression model {name!r}.")


def model_parameters(model: Any) -> dict[str, Any]:
    if hasattr(model, "get_params"):
        return {
            key: value for key, value in model.get_params(deep=True).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
    return {"class": type(model).__name__, "statistic": getattr(model, "statistic", None)}


def _correlation(first: np.ndarray, second: np.ndarray, rank: bool = False) -> float | None:
    if len(first) < 2:
        return None
    if rank:
        first = pd.Series(first).rank(method="average").to_numpy()
        second = pd.Series(second).rank(method="average").to_numpy()
    if np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """Primary V2 metrics; input and core errors are log10(brightness + 1)."""
    residual = prediction - y_true
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    raw_true, raw_prediction = inverse_transform_brightness(y_true), inverse_transform_brightness(prediction)
    return {
        "spearman_rho": _correlation(y_true, prediction, rank=True),
        "pearson_r": _correlation(y_true, prediction),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": None if total == 0 else float(1 - np.sum(residual ** 2) / total),
        "raw_rmse_supplementary": float(np.sqrt(np.mean((raw_prediction - raw_true) ** 2))),
        "raw_mae_supplementary": float(np.mean(np.abs(raw_prediction - raw_true))),
    }


def aggregate_regression_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = tuple(folds[0]["metrics"])
    result: dict[str, Any] = {"fold_count": len(folds)}
    for name in metric_names:
        values = [fold["metrics"][name] for fold in folds if fold["metrics"][name] is not None]
        result[name] = None if not values else {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    return result


def select_regressor(results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Select by pre-specified mean validation Spearman rho, then stability."""
    available = {
        name: value for name, value in results.items()
        if value["aggregate"]["spearman_rho"] is not None
    }
    if not available:
        raise ValueError("No model produced a defined Spearman rho across V2 folds.")
    selected = min(
        available,
        key=lambda name: (
            -available[name]["aggregate"]["spearman_rho"]["mean"],
            available[name]["aggregate"]["spearman_rho"]["std"],
            V2_MODEL_ORDER[name],
        ),
    )
    return selected, {
        "primary_criterion": "highest_mean_validation_spearman_rho",
        "tie_breakers": ["lower_spearman_sd", "prespecified_model_order"],
        "selected_model": selected,
    }


def _run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_regression_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = {"artifact_version", "model", "target_transformation", "feature_version", "metadata"}
    if not isinstance(artifact, dict) or required.difference(artifact):
        raise ValueError("V2 regression artifact is incomplete or incompatible.")
    if artifact["artifact_version"] != V2_ARTIFACT_VERSION:
        raise ValueError("V2 regression artifact version is incompatible.")
    if artifact["feature_version"] != CompositionFeaturizer.feature_version:
        raise ValueError("V2 regression artifact uses an incompatible feature version.")
    return artifact


def train_regression_experiment(
    data_path: str | Path,
    config: dict[str, Any],
    results_dir: str | Path = "results/v2_homology_regression",
    source_metadata: dict[str, Any] | None = None,
    cleaning_summary: dict[str, Any] | None = None,
) -> Path:
    """Run V2 alignment-cluster-held-out CV and serialize its selected model."""
    source = Path(data_path)
    frame = pd.read_csv(source)
    required = {"record_id", "sequence", "brightness"}
    missing = required.difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"V2 data must be nonempty and contain {sorted(required)}; missing={sorted(missing)}")
    records = frame[["record_id", "sequence"]].to_dict("records")
    homology = build_homology_clusters(records, float(config["homology"]["identity_threshold"]))
    groups = frame["record_id"].astype(str).map(homology.group_ids).to_numpy()
    if pd.isna(groups).any():
        raise RuntimeError("Every V2 observation must receive a homology cluster.")
    splits = list(grouped_folds(groups, int(config["split"]["folds"]), seed=int(config["split"]["seed"])))
    sequences = frame["sequence"].astype(str).to_numpy()
    brightness = frame["brightness"].astype(float).to_numpy()
    transformed = transform_brightness(brightness)
    model_results: dict[str, Any] = {}
    seed = int(config["split"]["seed"])
    for model_index, name in enumerate(config["models"]):
        folds: list[dict[str, Any]] = []
        for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
            model = make_regressor(name, seed + model_index * 100 + fold_number)
            model.fit(sequences[train_index], transformed[train_index])
            prediction = np.asarray(model.predict(sequences[validation_index]), dtype=float)
            folds.append({
                "fold": fold_number, "train_size": int(len(train_index)), "validation_size": int(len(validation_index)),
                "train_clusters": sorted(set(groups[train_index])), "validation_clusters": sorted(set(groups[validation_index])),
                "metrics": regression_metrics(transformed[validation_index], prediction),
            })
        model_results[name] = {
            "parameters": model_parameters(make_regressor(name, seed)), "folds": folds,
            "aggregate": aggregate_regression_metrics(folds),
        }
    selected_model, selection = select_regressor(model_results)
    final_model = make_regressor(selected_model, seed)
    final_model.fit(sequences, transformed)
    run_directory = Path(results_dir) / _run_id()
    run_directory.mkdir(parents=True, exist_ok=False)
    assignments = [
        {"fold": fold_number, "record_id": frame.iloc[index]["record_id"], "homology_cluster_id": groups[index]}
        for fold_number, (_, validation_index) in enumerate(splits, start=1) for index in validation_index
    ]
    pd.DataFrame(assignments).to_csv(run_directory / "fold_assignments.csv", index=False)
    pd.DataFrame(homology.edges, columns=["record_id_a", "record_id_b"]).to_csv(run_directory / "homology_edges.csv", index=False)
    metadata = {
        "experiment": "v2_homology_regression", "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "artifact_version": V2_ARTIFACT_VERSION, "source_data": str(source), "source_data_sha256": _hash_file(source),
        "source": source_metadata or {}, "cleaning_summary": cleaning_summary or {},
        "raw_records": int((source_metadata or {}).get("record_count", len(frame))), "cleaned_records": int(len(frame)),
        "scope": config["scope"], "state_policy": config["state_policy"], "target": config["target"],
        "target_transformation": "log10(brightness + 1)",
        "features": {"name": "composition", "version": CompositionFeaturizer.feature_version,
                     "names": CompositionFeaturizer().get_feature_names_out().tolist()},
        "homology": {**config["homology"], "alignment_method": ALIGNMENT_METHOD,
                     "alignment_scoring": ALIGNMENT_SCORING,
                     "identity_definition": IDENTITY_DEFINITION, "graph_sha256": homology.graph_sha256,
                     "cluster_summary": homology.summary},
        "split": config["split"], "selected_model": selected_model, "model_selection": selection,
        "python": sys.version, "platform": platform.platform(),
        "package_versions": {"joblib": joblib.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                             "scikit_learn": sklearn.__version__, "biopython": homology.summary["biopython_version"]},
    }
    artifact = {
        "artifact_version": V2_ARTIFACT_VERSION, "model": final_model,
        "target_transformation": "log10(brightness + 1)", "feature_version": CompositionFeaturizer.feature_version,
        "metadata": metadata,
    }
    joblib.dump(artifact, run_directory / "model.joblib")
    (run_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_directory / "metrics.json").write_text(json.dumps(model_results, indent=2), encoding="utf-8")
    (run_directory / "cleaning_summary.json").write_text(json.dumps(cleaning_summary or {}, indent=2), encoding="utf-8")
    load_regression_artifact(run_directory / "model.joblib")
    return run_directory


def train_representative_regression_experiment(
    data_path: str | Path,
    config: dict[str, Any],
    results_dir: str | Path = "results/v2b_representative_regression",
    source_metadata: dict[str, Any] | None = None,
    cleaning_summary: dict[str, Any] | None = None,
) -> Path:
    """Run V2b representative-cluster-held-out continuous-brightness CV."""
    source = Path(data_path)
    frame = pd.read_csv(source)
    required = {"record_id", "sequence", "brightness"}
    missing = required.difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"V2b data must be nonempty and contain {sorted(required)}; missing={sorted(missing)}")
    records = frame[["record_id", "sequence"]].to_dict("records")
    clusters = build_representative_clusters(records, float(config["homology"]["identity_threshold"]))
    groups = frame["record_id"].astype(str).map(clusters.group_ids).to_numpy()
    if pd.isna(groups).any() or len(set(groups)) < int(config["split"]["folds"]):
        raise RuntimeError("V2b does not have enough complete representative clusters for the configured CV.")
    splits = list(grouped_folds(groups, int(config["split"]["folds"]), seed=int(config["split"]["seed"])))
    sequences = frame["sequence"].astype(str).to_numpy()
    transformed = transform_brightness(frame["brightness"].astype(float).to_numpy())
    seed = int(config["split"]["seed"])
    model_results: dict[str, Any] = {}
    for model_index, name in enumerate(config["models"]):
        folds: list[dict[str, Any]] = []
        for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
            model = make_regressor(name, seed + model_index * 100 + fold_number)
            model.fit(sequences[train_index], transformed[train_index])
            prediction = np.asarray(model.predict(sequences[validation_index]), dtype=float)
            overlap = set(groups[train_index]).intersection(groups[validation_index])
            if overlap:
                raise RuntimeError(f"V2b representative cluster overlap in fold {fold_number}: {sorted(overlap)}")
            folds.append({
                "fold": fold_number, "train_size": int(len(train_index)), "validation_size": int(len(validation_index)),
                "train_clusters": sorted(set(groups[train_index])), "validation_clusters": sorted(set(groups[validation_index])),
                "metrics": regression_metrics(transformed[validation_index], prediction),
            })
        model_results[name] = {
            "parameters": model_parameters(make_regressor(name, seed)), "folds": folds,
            "aggregate": aggregate_regression_metrics(folds),
        }
    selected_model, selection = select_regressor(model_results)
    final_model = make_regressor(selected_model, seed).fit(sequences, transformed)
    run_directory = Path(results_dir) / _run_id()
    run_directory.mkdir(parents=True, exist_ok=False)
    record_to_fold = {
        int(index): fold_number
        for fold_number, (_, validation_index) in enumerate(splits, start=1)
        for index in validation_index
    }
    assignments = pd.DataFrame({
        "fold": [record_to_fold[index] for index in range(len(frame))],
        "record_id": frame["record_id"].astype(str),
        "representative_cluster_id": groups,
        "representative_record_id": [clusters.group_ids[str(record_id)].removeprefix("cluster:") for record_id in frame["record_id"]],
    })
    assignments.to_csv(run_directory / "fold_assignments.csv", index=False)
    assignments.assign(
        representative_identity=[clusters.representative_identity[str(record_id)] for record_id in frame["record_id"]]
    ).to_csv(run_directory / "cluster_assignments.csv", index=False)
    metadata = {
        "experiment": "v2b_representative_regression", "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "artifact_version": V2_ARTIFACT_VERSION, "source_data": str(source), "source_data_sha256": _hash_file(source),
        "source": source_metadata or {}, "cleaning_summary": cleaning_summary or {},
        "raw_records": int((source_metadata or {}).get("record_count", len(frame))), "cleaned_records": int(len(frame)),
        "scope": config["scope"], "state_policy": config["state_policy"], "target": config["target"],
        "target_transformation": "log10(brightness + 1)",
        "features": {"name": "composition", "version": CompositionFeaturizer.feature_version,
                     "names": CompositionFeaturizer().get_feature_names_out().tolist()},
        "clustering": {**config["homology"], "alignment_method": ALIGNMENT_METHOD,
                       "alignment_scoring": ALIGNMENT_SCORING, "identity_definition": IDENTITY_DEFINITION,
                       "assignment_sha256": clusters.assignment_sha256, "cluster_summary": clusters.summary},
        "split": config["split"], "selected_model": selected_model, "model_selection": selection,
        "python": sys.version, "platform": platform.platform(),
        "package_versions": {"joblib": joblib.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                             "scikit_learn": sklearn.__version__, "biopython": clusters.summary["biopython_version"]},
    }
    artifact = {
        "artifact_version": V2_ARTIFACT_VERSION, "model": final_model,
        "target_transformation": "log10(brightness + 1)", "feature_version": CompositionFeaturizer.feature_version,
        "metadata": metadata,
    }
    joblib.dump(artifact, run_directory / "model.joblib")
    (run_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_directory / "metrics.json").write_text(json.dumps(model_results, indent=2), encoding="utf-8")
    (run_directory / "cleaning_summary.json").write_text(json.dumps(cleaning_summary or {}, indent=2), encoding="utf-8")
    load_regression_artifact(run_directory / "model.joblib")
    return run_directory
