"""Group-preserving cross-validation helpers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
class SplitError(ValueError):
    """A requested evaluation split would violate grouping requirements."""


def grouped_folds(groups, n_splits: int, seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield seeded, sample-count-balanced, group-disjoint folds.

    Fold construction deliberately uses only group membership and group size,
    never brightness or tier labels.  This avoids using held-out target values
    to engineer a more balanced split while still making the configured seed
    meaningful and the assignment reproducible across scikit-learn versions.
    """
    groups_array = np.asarray(groups, dtype=object)
    unique = np.unique(groups_array)
    if len(unique) < n_splits:
        raise SplitError(
            f"Requested {n_splits} folds but only {len(unique)} leakage-control groups are available."
        )
    counts = {group: int(np.sum(groups_array == group)) for group in unique}
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(unique))
    shuffled_rank = {group: rank for rank, group in enumerate(shuffled)}
    # Largest groups first is the usual greedy bin-packing safeguard.  The
    # seeded rank resolves ties, including the common singleton-group case.
    ordered_groups = sorted(unique, key=lambda group: (-counts[group], shuffled_rank[group]))
    fold_groups: list[list[object]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for group in ordered_groups:
        candidate_folds = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold = int(rng.choice(candidate_folds))
        fold_groups[fold].append(group)
        fold_sizes[fold] += counts[group]

    for validation_group_values in fold_groups:
        validation_mask = np.isin(groups_array, validation_group_values)
        validation_index = np.flatnonzero(validation_mask)
        train_index = np.flatnonzero(~validation_mask)
        overlap = set(groups_array[train_index]).intersection(groups_array[validation_index])
        if overlap:
            raise SplitError(f"Group overlap detected: {sorted(overlap)}")
        yield train_index, validation_index
