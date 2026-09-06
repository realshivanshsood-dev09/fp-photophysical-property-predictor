"""Conservative baselines and small-data sklearn model factories."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import CompositionFeaturizer


class MajorityBaseline:
    def fit(self, X, y):
        counts = np.bincount(np.asarray(y, dtype=int), minlength=3)
        self.class_ = int(np.argmax(counts))
        self.class_probabilities_ = counts / counts.sum()
        self.classes_ = np.arange(3)
        return self

    def predict(self, X):
        return np.full(len(X), self.class_, dtype=int)

    def predict_proba(self, X):
        return np.tile(self.class_probabilities_, (len(X), 1))


class StratifiedRandomBaseline:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, X, y):
        counts = np.bincount(np.asarray(y, dtype=int), minlength=3)
        self.class_probabilities_ = counts / counts.sum()
        self.classes_ = np.arange(3)
        return self

    def predict(self, X):
        # A fixed per-fold generator makes the benchmark reproducible and uses
        # the training labels only.
        return np.random.default_rng(self.seed).choice(3, size=len(X), p=self.class_probabilities_)

    def predict_proba(self, X):
        return np.tile(self.class_probabilities_, (len(X), 1))


def make_model(name: str, seed: int) -> Any:
    if name == "majority":
        return MajorityBaseline()
    if name == "stratified_random":
        return StratifiedRandomBaseline(seed=seed)
    if name == "logistic_regression":
        return Pipeline([
            ("features", CompositionFeaturizer()),
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("features", CompositionFeaturizer()),
            ("classifier", RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=seed, n_jobs=1,
            )),
        ])
    if name == "gradient_boosting":
        return Pipeline([
            ("features", CompositionFeaturizer()),
            ("classifier", GradientBoostingClassifier(
                n_estimators=80, learning_rate=0.05, max_depth=2, min_samples_leaf=2, random_state=seed,
            )),
        ])
    raise ValueError(f"Unsupported model {name!r}.")


def model_parameters(model: Any) -> dict[str, Any]:
    if hasattr(model, "get_params"):
        return {key: value for key, value in model.get_params(deep=True).items() if isinstance(value, (str, int, float, bool, type(None)))}
    return {"class": type(model).__name__, "seed": getattr(model, "seed", None)}


def predict_proba_aligned(model: Any, X) -> np.ndarray:
    """Always emit LOW/MEDIUM/HIGH columns even if a fold lacked a class."""
    probability = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_", np.arange(probability.shape[1])), dtype=int)
    aligned = np.zeros((len(X), 3), dtype=float)
    aligned[:, classes] = probability
    return aligned
