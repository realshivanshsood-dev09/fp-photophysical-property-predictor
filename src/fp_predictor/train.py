"""Leakage-safe training, evaluation, metadata, and deployable artifact creation."""

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

from .evaluate import CLASS_NAMES, aggregate_fold_metrics, classification_metrics
from .features import CompositionFeaturizer
from .models import make_model, model_parameters, predict_proba_aligned
from .split import grouped_folds

ARTIFACT_VERSION = "1"
MODEL_COMPLEXITY_ORDER = {
    "majority": 0, "stratified_random": 0, "logistic_regression": 1,
    "gradient_boosting": 2, "random_forest": 3,
}
LEARNED_MODELS = frozenset({"logistic_regression", "random_forest", "gradient_boosting"})


def derive_thresholds(brightness) -> list[float]:
    values = np.asarray(brightness, dtype=float)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("At least three finite brightness values are required for tertile thresholds.")
    thresholds = np.quantile(values, [1 / 3, 2 / 3], method="linear")
    if thresholds[0] >= thresholds[1]:
        raise ValueError("Brightness tertiles collapse; three operational classes are not identifiable.")
    return [float(value) for value in thresholds]


def apply_thresholds(brightness, thresholds: list[float]) -> np.ndarray:
    if len(thresholds) != 2 or thresholds[0] >= thresholds[1]:
        raise ValueError("Expected two strictly increasing thresholds.")
    return np.digitize(np.asarray(brightness, dtype=float), bins=np.asarray(thresholds), right=False).astype(int)


def _run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_model(all_models: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Use full fold summaries, with a simplicity tie-break inside one SD.

    The rule deliberately does not select on a best individual fold.  Models
    within one standard deviation of the top mean macro-F1 are treated as
    practically indistinguishable for this exploratory benchmark; the simpler,
    then more stable, model wins among them.
    """
    learned = {name: value for name, value in all_models.items() if name in LEARNED_MODELS}
    if not learned:
        raise ValueError("At least one learned classifier is required for a deployable model artifact.")
    top_mean = max(value["aggregate"]["macro_f1"]["mean"] for value in learned.values())
    top_models = [
        name for name, value in learned.items()
        if value["aggregate"]["macro_f1"]["mean"] == top_mean
    ]
    top_sd = min(all_models[name]["aggregate"]["macro_f1"]["std"] for name in top_models)
    candidates = [
        name for name, value in learned.items()
        if value["aggregate"]["macro_f1"]["mean"] >= top_mean - top_sd
    ]
    selected = min(
        candidates,
        key=lambda name: (
            MODEL_COMPLEXITY_ORDER.get(name, 99),
            all_models[name]["aggregate"]["macro_f1"]["std"],
            -all_models[name]["aggregate"]["macro_f1"]["mean"],
            name,
        ),
    )
    return selected, {
        "policy": "one_standard_deviation_simplicity",
        "top_mean_macro_f1": top_mean,
        "top_model_sd": top_sd,
        "candidates_within_top_model_sd": candidates,
        "selected_model": selected,
    }


def dataset_summary(frame: pd.DataFrame, production_thresholds: list[float]) -> dict[str, Any]:
    labels = apply_thresholds(frame["brightness"].astype(float).to_numpy(), production_thresholds)
    group_sizes = frame.groupby("group_id", dropna=False).size()
    summary: dict[str, Any] = {
        "observations": int(len(frame)),
        "unique_sequences": int(frame["sequence"].nunique()),
        "leakage_control_groups": int(frame["group_id"].nunique()),
        "group_size": {
            "min": int(group_sizes.min()), "max": int(group_sizes.max()),
            "median": float(group_sizes.median()), "groups_larger_than_one": int((group_sizes > 1).sum()),
        },
        "production_class_distribution": {
            CLASS_NAMES[index]: int((labels == index).sum()) for index in range(len(CLASS_NAMES))
        },
        "feature_count": int(len(CompositionFeaturizer().get_feature_names_out())),
        "feature_names": CompositionFeaturizer().get_feature_names_out().tolist(),
    }
    for column in ("cofactor", "switch_type", "chromophore"):
        if column in frame:
            values = frame[column].fillna("UNSPECIFIED").astype(str).value_counts().to_dict()
            summary[f"{column}_composition"] = {str(key): int(value) for key, value in values.items()}
    return summary


def train_experiment(
    data_path: str | Path,
    config: dict[str, Any],
    results_dir: str | Path = "results",
    source_metadata: dict[str, Any] | None = None,
    cleaning_summary: dict[str, Any] | None = None,
) -> Path:
    """Evaluate every configured model, then fit and serialize the best full-data model."""
    source = Path(data_path)
    frame = pd.read_csv(source)
    required = {"record_id", "sequence", "brightness", "group_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed data lacks required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Processed data is empty.")
    groups = frame["group_id"].astype(str).to_numpy()
    sequences = frame["sequence"].astype(str).to_numpy()
    brightness = frame["brightness"].astype(float).to_numpy()
    splits = list(grouped_folds(groups, int(config["split"]["folds"]), seed=int(config["split"]["seed"])))
    all_models: dict[str, Any] = {}
    seed = int(config["split"]["seed"])
    for model_index, name in enumerate(config["models"]):
        folds: list[dict[str, Any]] = []
        for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
            thresholds = derive_thresholds(brightness[train_index])
            y_train = apply_thresholds(brightness[train_index], thresholds)
            y_validation = apply_thresholds(brightness[validation_index], thresholds)
            if len(np.unique(y_train)) < 3:
                raise ValueError(f"Fold {fold_number} has fewer than three training classes after thresholding.")
            model = make_model(name, seed=seed + model_index * 100 + fold_number)
            model.fit(sequences[train_index], y_train)
            prediction = np.asarray(model.predict(sequences[validation_index]), dtype=int)
            metrics = classification_metrics(y_validation, prediction)
            folds.append({
                "fold": fold_number,
                "thresholds": thresholds,
                "train_size": int(len(train_index)), "validation_size": int(len(validation_index)),
                "train_groups": sorted(set(groups[train_index])),
                "validation_groups": sorted(set(groups[validation_index])),
                "metrics": metrics,
            })
        all_models[name] = {
            "parameters": model_parameters(make_model(name, seed)),
            "folds": folds,
            "aggregate": aggregate_fold_metrics(folds),
        }
    best_name, selection = select_model(all_models)
    production_thresholds = derive_thresholds(brightness)
    production_y = apply_thresholds(brightness, production_thresholds)
    final_model = make_model(best_name, seed=seed)
    final_model.fit(sequences, production_y)

    root = Path(results_dir)
    run_directory = root / _run_id()
    counter = 1
    while run_directory.exists():
        run_directory = root / f"{_run_id()}_{counter}"
        counter += 1
    run_directory.mkdir(parents=True, exist_ok=False)
    assignments: list[dict[str, Any]] = []
    for fold_number, (train_index, validation_index) in enumerate(splits, start=1):
        for index in validation_index:
            assignments.append({"fold": fold_number, "record_id": frame.iloc[index]["record_id"], "group_id": groups[index]})
    pd.DataFrame(assignments).to_csv(run_directory / "fold_assignments.csv", index=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "artifact_version": ARTIFACT_VERSION,
        "source_data": str(source), "source_data_sha256": _hash_file(source),
        "source": source_metadata or {},
        "cleaning_summary": cleaning_summary or {},
        "raw_records": int((source_metadata or {}).get("record_count", len(frame))),
        "cleaned_records": int(len(frame)), "unique_groups": int(len(set(groups))),
        "config": config,
        "scope": config["scope"], "state_policy": config["state_policy"],
        "grouping_source": sorted(frame["grouping_source"].dropna().astype(str).unique().tolist()) if "grouping_source" in frame else ["unspecified"],
        "target": config["target"], "features": {"name": "composition", "version": CompositionFeaturizer.feature_version},
        "split": config["split"], "class_names": list(CLASS_NAMES),
        "production_thresholds": production_thresholds, "selected_model": best_name,
        "model_selection": selection,
        "python": sys.version, "platform": platform.platform(),
        "package_versions": {
            "joblib": joblib.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
    }
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model": final_model,
        "class_names": list(CLASS_NAMES),
        "thresholds": production_thresholds,
        "feature_name": "composition", "feature_version": CompositionFeaturizer.feature_version,
        "metadata": metadata,
    }
    joblib.dump(artifact, run_directory / "model.joblib")
    data_summary = dataset_summary(frame, production_thresholds)
    (run_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_directory / "cleaning_summary.json").write_text(
        json.dumps(cleaning_summary or {}, indent=2), encoding="utf-8"
    )
    (run_directory / "dataset_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")
    (run_directory / "metrics.json").write_text(json.dumps(all_models, indent=2), encoding="utf-8")
    matrices = {name: [fold["metrics"]["confusion_matrix"] for fold in model["folds"]] for name, model in all_models.items()}
    (run_directory / "confusion_matrices.json").write_text(json.dumps(matrices, indent=2), encoding="utf-8")
    return run_directory
