from __future__ import annotations

import numpy as np

from fp_predictor.features import CompositionFeaturizer
from fp_predictor.split import grouped_folds


def test_composition_features_are_deterministic_and_sequence_only():
    featurizer = CompositionFeaturizer().fit(["MACKRDEFGH"])
    first = featurizer.transform(["MACKRDEFGH"])
    second = featurizer.transform(["MACKRDEFGH"])
    assert first.shape == (1, 39)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_grouped_folds_have_no_group_overlap():
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
    for train, validation in grouped_folds(groups, 5):
        assert not set(groups[train]).intersection(groups[validation])


def test_grouped_folds_are_seeded_and_reproducible():
    groups = np.array([f"g{index}" for index in range(10)])
    first = [tuple(validation) for _, validation in grouped_folds(groups, 5, seed=7)]
    second = [tuple(validation) for _, validation in grouped_folds(groups, 5, seed=7)]
    changed = [tuple(validation) for _, validation in grouped_folds(groups, 5, seed=8)]
    assert first == second
    assert first != changed
