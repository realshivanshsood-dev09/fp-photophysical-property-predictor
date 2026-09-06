"""Model-artifact validation and strict multi-FASTA prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .clean import CleaningError, normalize_sequence
from .evaluate import CLASS_NAMES
from .features import CompositionFeaturizer
from .models import predict_proba_aligned
from .train import ARTIFACT_VERSION


class PredictionError(ValueError):
    """A FASTA input or serialized artifact is unsafe to use."""


def parse_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Parse a strict multi-FASTA file, rejecting ambiguous or malformed input."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PredictionError(f"Cannot read FASTA file {source}: {error}") from error
    entries: list[tuple[str, str]] = []
    name: str | None = None
    fragments: list[str] = []
    names: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                try:
                    entries.append((name, normalize_sequence("".join(fragments))))
                except CleaningError as error:
                    raise PredictionError(f"Invalid FASTA record {name!r}: {error}") from error
            name = line[1:].strip()
            if not name:
                raise PredictionError(f"FASTA header is empty at line {line_number}.")
            if name in names:
                raise PredictionError(f"Duplicate FASTA header {name!r}.")
            names.add(name)
            fragments = []
        elif name is None:
            raise PredictionError(f"Sequence precedes a FASTA header at line {line_number}.")
        else:
            fragments.append(line)
    if name is not None:
        try:
            entries.append((name, normalize_sequence("".join(fragments))))
        except CleaningError as error:
            raise PredictionError(f"Invalid FASTA record {name!r}: {error}") from error
    if not entries:
        raise PredictionError("No FASTA records found.")
    return entries


def load_artifact(path: str | Path) -> dict:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise PredictionError(f"Trained model not found: {artifact_path}")
    try:
        artifact = joblib.load(artifact_path)
    except Exception as error:  # joblib can surface varied serialization exceptions
        raise PredictionError(f"Could not load model artifact: {error}") from error
    required = {"artifact_version", "model", "class_names", "thresholds", "feature_version"}
    if not isinstance(artifact, dict) or required.difference(artifact):
        raise PredictionError("Model artifact is incomplete or incompatible.")
    if artifact["artifact_version"] != ARTIFACT_VERSION:
        raise PredictionError("Model artifact version is incompatible with this CLI.")
    if artifact["feature_version"] != CompositionFeaturizer.feature_version:
        raise PredictionError("Model artifact uses an incompatible feature version.")
    if artifact["class_names"] != list(CLASS_NAMES):
        raise PredictionError("Model artifact has an incompatible class mapping.")
    return artifact


def predict_fasta(fasta_path: str | Path, model_path: str | Path) -> pd.DataFrame:
    entries = parse_fasta(fasta_path)
    artifact = load_artifact(model_path)
    names, sequences = zip(*entries)
    prediction = np.asarray(artifact["model"].predict(list(sequences)), dtype=int)
    probabilities = predict_proba_aligned(artifact["model"], list(sequences))
    result = pd.DataFrame({"protein": names, "predicted_brightness_tier": [CLASS_NAMES[item] for item in prediction]})
    for index, class_name in enumerate(CLASS_NAMES):
        result[f"probability_{class_name.lower()}"] = probabilities[:, index]
    result["model"] = artifact["metadata"].get("selected_model", type(artifact["model"]).__name__)
    result["feature_set"] = artifact["feature_name"]
    return result
