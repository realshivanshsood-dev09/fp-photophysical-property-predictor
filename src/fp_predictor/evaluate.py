"""Fold-level classification metrics and JSON-safe summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


CLASS_NAMES = ("LOW", "MEDIUM", "HIGH")


def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    labels = np.arange(len(CLASS_NAMES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    # Grouped small-N folds can legitimately omit a class in validation.  Use
    # the usual mean recall over classes represented in that fold, rather than
    # asking sklearn to warn about predictions in a class absent from y_true.
    present_labels = np.unique(np.asarray(y_true, dtype=int))
    return {
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall[present_labels])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_precision": {name: float(value) for name, value in zip(CLASS_NAMES, precision)},
        "per_class_recall": {name: float(value) for name, value in zip(CLASS_NAMES, recall)},
        "per_class_f1": {name: float(value) for name, value in zip(CLASS_NAMES, f1)},
        "support": {name: int(value) for name, value in zip(CLASS_NAMES, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def aggregate_fold_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    if not folds:
        raise ValueError("Cannot aggregate zero folds.")
    metrics = ("macro_f1", "balanced_accuracy", "accuracy")
    result = {
        metric: {
            "mean": float(np.mean([fold["metrics"][metric] for fold in folds])),
            "std": float(np.std([fold["metrics"][metric] for fold in folds], ddof=0)),
        }
        for metric in metrics
    }
    result["fold_count"] = len(folds)
    for metric_name in ("per_class_precision", "per_class_recall", "per_class_f1"):
        result[metric_name] = {
            class_name: {
                "mean": float(np.mean([fold["metrics"][metric_name][class_name] for fold in folds])),
                "std": float(np.std([fold["metrics"][metric_name][class_name] for fold in folds], ddof=0)),
            }
            for class_name in CLASS_NAMES
        }
    result["aggregate_confusion_matrix"] = np.sum(
        [np.asarray(fold["metrics"]["confusion_matrix"], dtype=int) for fold in folds], axis=0
    ).astype(int).tolist()
    return result
