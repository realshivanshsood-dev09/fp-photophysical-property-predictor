"""Lineage utilities and conservative fallback grouping.

FPbase documents a recursive lineage relation, but the public GraphQL response
used by :mod:`fp_predictor.fetch` does not expose those edges.  A user may pass
a sidecar of ``child_id,parent_id`` rows exported from a verified source.  This
module never infers ancestry from organism or protein names.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def load_lineage_sidecar(path: str | Path) -> dict[str, str]:
    """Load a directed child-to-parent CSV and reject ambiguous parentage."""
    parents: dict[str, str] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"child_id", "parent_id"}.issubset(reader.fieldnames):
            raise ValueError("Lineage sidecar must have child_id,parent_id columns.")
        for row in reader:
            child = (row.get("child_id") or "").strip()
            parent = (row.get("parent_id") or "").strip()
            if not child or not parent:
                raise ValueError("Lineage sidecar cannot contain blank child_id or parent_id.")
            if child in parents and parents[child] != parent:
                raise ValueError(f"Multiple parents for {child!r}; v1 needs an unambiguous directed ancestry.")
            parents[child] = parent
    return parents


def lineage_roots(record_ids: Iterable[str], parents: dict[str, str]) -> dict[str, str]:
    """Map records to root tokens, safely handling missing ancestors and cycles.

    An unknown parent groups all children carrying that same unknown parent token.
    A cycle is assigned to a canonical cycle token rather than recursing forever.
    """
    known = {str(value) for value in record_ids}
    result: dict[str, str] = {}

    def resolve(start: str) -> str:
        if start in result:
            return result[start]
        path: list[str] = []
        position: dict[str, int] = {}
        current = start
        while True:
            if current in result:
                root = result[current]
                break
            if current in position:
                cycle = path[position[current] :]
                root = "cycle:" + min(cycle)
                for item in cycle:
                    result[item] = root
                break
            position[current] = len(path)
            path.append(current)
            parent = parents.get(current)
            if parent is None:
                root = current
                break
            if parent not in known and parent not in parents:
                root = "missing-parent:" + parent
                break
            current = parent
        for item in path:
            result.setdefault(item, root)
        return result[start]

    return {identifier: resolve(identifier) for identifier in known}


class _DisjointSet:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = self.parent[item]
        while root != self.parent[root]:
            root = self.parent[root]
        while item != root:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[max(first_root, second_root)] = min(first_root, second_root)


def near_duplicate_pairs(records: list[dict], identity_threshold: float | None) -> list[tuple[str, str]]:
    """Return conservative same-length sequence-neighbour pairs.

    This is deliberately not an alignment or an inferred lineage.  It catches
    ordinary point-mutant series by requiring positional identity across equal
    length sequences; indel-containing relatives remain a documented residual
    leakage risk.  The O(N²L) implementation is intentional for this small
    dataset and avoids introducing an opaque clustering dependency.
    """
    if identity_threshold is None:
        return []
    if not 0 < identity_threshold < 1:
        raise ValueError("near_duplicate_identity must be strictly between 0 and 1.")
    pairs: list[tuple[str, str]] = []
    for index, record in enumerate(records):
        identifier = str(record["record_id"])
        sequence = str(record["sequence"])
        for earlier in records[:index]:
            earlier_identifier = str(earlier["record_id"])
            earlier_sequence = str(earlier["sequence"])
            if identifier == earlier_identifier or len(sequence) != len(earlier_sequence):
                continue
            identity = sum(left == right for left, right in zip(sequence, earlier_sequence)) / len(sequence)
            if identity >= identity_threshold:
                pairs.append((identifier, earlier_identifier))
    return pairs


def build_group_ids(
    records: list[dict],
    parents: dict[str, str] | None = None,
    near_duplicate_identity: float | None = None,
) -> dict[str, str]:
    """Combine exact identity, optional sequence neighbours, and verified ancestry."""
    identifiers = [str(record["record_id"]) for record in records]
    dsu = _DisjointSet(identifiers)
    by_sequence: dict[str, str] = {}
    for record in records:
        identifier = str(record["record_id"])
        sequence = str(record["sequence"])
        if sequence in by_sequence:
            dsu.union(identifier, by_sequence[sequence])
        else:
            by_sequence[sequence] = identifier
    for first, second in near_duplicate_pairs(records, near_duplicate_identity):
        dsu.union(first, second)
    if parents:
        roots = lineage_roots(identifiers, parents)
        by_root: dict[str, str] = {}
        for identifier, root in roots.items():
            if root in by_root:
                dsu.union(identifier, by_root[root])
            else:
                by_root[root] = identifier
    members: dict[str, list[str]] = {}
    for identifier in identifiers:
        members.setdefault(dsu.find(identifier), []).append(identifier)
    canonical = {root: "group:" + min(items) for root, items in members.items()}
    return {identifier: canonical[dsu.find(identifier)] for identifier in identifiers}
