"""Compact, deterministic amino-acid composition and physicochemical features."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from fp_predictor.clean import normalize_sequence

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
RESIDUE_MASS = {
    "A": 71.0788, "C": 103.1388, "D": 115.0886, "E": 129.1155, "F": 147.1766,
    "G": 57.0519, "H": 137.1411, "I": 113.1594, "K": 128.1741, "L": 113.1594,
    "M": 131.1926, "N": 114.1038, "P": 97.1167, "Q": 128.1307, "R": 156.1875,
    "S": 87.0782, "T": 101.1051, "V": 99.1326, "W": 186.2132, "Y": 163.1760,
}


def _iter_sequences(X: Iterable[str]) -> list[str]:
    if hasattr(X, "columns") and "sequence" in X.columns:  # pandas frame
        X = X["sequence"].tolist()
    elif isinstance(X, np.ndarray):
        X = X.ravel().tolist()
    return [normalize_sequence(sequence) for sequence in X]


def _charge_at_ph(sequence: str, ph: float) -> float:
    # Henderson-Hasselbalch proxy, pKa values from standard educational tables.
    positive = 1 / (1 + 10 ** (ph - 9.69)) + 1 / (1 + 10 ** (ph - 8.0))
    negative = 1 / (1 + 10 ** (2.34 - ph)) + 1 / (1 + 10 ** (3.1 - ph))
    positive += sequence.count("K") / (1 + 10 ** (ph - 10.5))
    positive += sequence.count("R") / (1 + 10 ** (ph - 12.4))
    positive += sequence.count("H") / (1 + 10 ** (ph - 6.0))
    negative += sequence.count("D") / (1 + 10 ** (3.9 - ph))
    negative += sequence.count("E") / (1 + 10 ** (4.1 - ph))
    negative += sequence.count("C") / (1 + 10 ** (8.3 - ph))
    negative += sequence.count("Y") / (1 + 10 ** (10.1 - ph))
    return positive - negative


def _estimated_pi(sequence: str) -> float:
    low, high = 0.0, 14.0
    for _ in range(40):
        middle = (low + high) / 2
        if _charge_at_ph(sequence, middle) > 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


class CompositionFeaturizer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible sequence-only descriptors; no FPbase measurements enter X."""

    feature_version = "1.0"

    def fit(self, X: Iterable[str], y=None):  # noqa: D401 - sklearn signature
        self.feature_names_in_ = np.array(self.get_feature_names_out(), dtype=object)
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = [f"aa_fraction_{residue}" for residue in AMINO_ACIDS]
        names += [
            "length", "acidic_fraction", "basic_fraction", "charged_fraction",
            "net_charge_ph7", "estimated_pi", "hydropathy_mean", "hydropathy_std",
            "hydrophobic_fraction", "polar_fraction", "nonpolar_fraction", "aromatic_fraction",
            "glycine_fraction", "proline_fraction", "cysteine_fraction", "aliphatic_index",
            "molecular_weight", "sequence_entropy", "residue_diversity",
        ]
        return np.asarray(names, dtype=object)

    def transform(self, X: Iterable[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for sequence in _iter_sequences(X):
            length = len(sequence)
            fraction = lambda residues: sum(sequence.count(item) for item in residues) / length
            composition = [sequence.count(residue) / length for residue in AMINO_ACIDS]
            hydropathy = np.asarray([KYTE_DOOLITTLE[residue] for residue in sequence], dtype=float)
            probabilities = np.asarray(composition, dtype=float)
            nonzero = probabilities[probabilities > 0]
            entropy = float(-(nonzero * np.log2(nonzero)).sum())
            rows.append(composition + [
                float(length), fraction("DE"), fraction("KR"), fraction("DEKR"),
                _charge_at_ph(sequence, 7.0), _estimated_pi(sequence),
                float(hydropathy.mean()), float(hydropathy.std()), fraction("AVILMFWY"),
                fraction("STNQCY"), fraction("AGILMFWVP"), fraction("FWY"),
                fraction("G"), fraction("P"), fraction("C"),
                100 * (fraction("A") + 2.9 * fraction("V") + 3.9 * fraction("IL")),
                float(sum(RESIDUE_MASS[item] for item in sequence) + 18.015), entropy,
                float((probabilities > 0).sum() / len(AMINO_ACIDS)),
            ])
        return np.asarray(rows, dtype=float)
