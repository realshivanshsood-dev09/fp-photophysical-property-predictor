"""Alignment-based sequence-similarity clustering for the V2 experiment.

This module is deliberately separate from V1's same-length near-duplicate
grouping.  It does not infer engineering lineage or use target values.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from Bio import __version__ as BIOPYTHON_VERSION
from Bio.Align import PairwiseAligner

from .lineage import _DisjointSet

ALIGNMENT_METHOD = "needleman_wunsch_global_linear_gap"
IDENTITY_DEFINITION = "identical_residue_columns / all_global_alignment_columns_including_gaps"
ALIGNMENT_SCORING = {
    "match_score": 1.0,
    "mismatch_score": -1.0,
    "open_gap_score": -1.0,
    "extend_gap_score": -1.0,
}


def _aligner() -> PairwiseAligner:
    """Build the fixed global Needleman--Wunsch aligner used by V2.

    Match = +1, mismatch = -1, and every gap residue = -1.  The first global
    alignment returned by Biopython is used; its version is saved in the run
    metadata.  This makes the procedure deterministic for a fixed runtime.
    """
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = ALIGNMENT_SCORING["match_score"]
    aligner.mismatch_score = ALIGNMENT_SCORING["mismatch_score"]
    aligner.open_gap_score = ALIGNMENT_SCORING["open_gap_score"]
    aligner.extend_gap_score = ALIGNMENT_SCORING["extend_gap_score"]
    return aligner


_GLOBAL_ALIGNER = _aligner()


def global_alignment_identity(first: str, second: str) -> float:
    """Return V2 global-alignment identity, with gaps counted in denominator.

    The denominator is the number of columns in the chosen global alignment:
    residue-residue columns plus residue-gap columns.  Consequently an indel
    lowers identity rather than being ignored.  This is an operational metric,
    not a universal biological definition of protein homology.
    """
    if not first or not second:
        raise ValueError("Alignment identity requires non-empty sequences.")
    if first == second:
        return 1.0
    alignment = _GLOBAL_ALIGNER.align(first, second)[0]
    coordinates = alignment.coordinates
    identical = 0
    columns = 0
    for index in range(coordinates.shape[1] - 1):
        first_start, first_end = (int(coordinates[0, index]), int(coordinates[0, index + 1]))
        second_start, second_end = (int(coordinates[1, index]), int(coordinates[1, index + 1]))
        first_width, second_width = first_end - first_start, second_end - second_start
        columns += max(first_width, second_width)
        if first_width and second_width:
            if first_width != second_width:
                raise RuntimeError("Unexpected non-diagonal residue-residue alignment segment.")
            identical += sum(
                left == right for left, right in zip(first[first_start:first_end], second[second_start:second_end])
            )
    if not columns:
        raise RuntimeError("Global alignment returned no columns.")
    return identical / columns


def homology_edges(records: list[dict], identity_threshold: float = 0.70) -> list[tuple[str, str]]:
    """Return deterministic record-ID edges at the configured alignment identity.

    A length-ratio screen is exact for this identity definition: because the
    alignment denominator is at least the longer sequence length, identity
    cannot exceed ``min_length / max_length``.  It only avoids impossible
    alignments and never uses sequences' targets or features.
    """
    if not 0 < identity_threshold <= 1:
        raise ValueError("identity_threshold must be in (0, 1].")
    representatives: dict[str, str] = {}
    for record in records:
        identifier, sequence = str(record["record_id"]), str(record["sequence"])
        representatives.setdefault(sequence, identifier)
    edges: list[tuple[str, str]] = []
    ordered_sequences = sorted(representatives)
    for first, second in combinations(ordered_sequences, 2):
        shorter, longer = sorted((len(first), len(second)))
        if shorter / longer < identity_threshold:
            continue
        # A second exact upper bound avoids most needless O(L²) alignments.
        # No alignment can have more identical residue columns than the sum of
        # per-residue multiset intersections, and its denominator cannot be
        # shorter than the longer input sequence.
        shared_residue_upper_bound = sum((Counter(first) & Counter(second)).values()) / longer
        if shared_residue_upper_bound < identity_threshold:
            continue
        # With the fixed +1/-1 Needleman--Wunsch scoring, an alignment whose
        # identity is at least t has score 2*matches-columns >= (2t-1)*L,
        # where L is at least the longer input length.  A lower optimal score
        # therefore proves that no qualifying alignment exists.  ``score`` is
        # materially faster than generating a traceback and is a safe screen.
        if _GLOBAL_ALIGNER.score(first, second) < (2 * identity_threshold - 1) * longer:
            continue
        if global_alignment_identity(first, second) >= identity_threshold:
            edges.append(tuple(sorted((representatives[first], representatives[second]))))
    return sorted(edges)


@dataclass(frozen=True)
class HomologyClusters:
    group_ids: dict[str, str]
    edges: list[tuple[str, str]]
    graph_sha256: str
    summary: dict[str, int | float]


def build_homology_clusters(records: list[dict], identity_threshold: float = 0.70) -> HomologyClusters:
    """Build transitive, target-independent connected components for V2."""
    identifiers = sorted({str(record["record_id"]) for record in records})
    if not identifiers:
        raise ValueError("Cannot cluster an empty record collection.")
    dsu = _DisjointSet(identifiers)
    by_sequence: dict[str, str] = {}
    for record in records:
        identifier, sequence = str(record["record_id"]), str(record["sequence"])
        if sequence in by_sequence:
            dsu.union(identifier, by_sequence[sequence])
        else:
            by_sequence[sequence] = identifier
    edges = homology_edges(records, identity_threshold)
    for first, second in edges:
        dsu.union(first, second)
    members: dict[str, list[str]] = {}
    for identifier in identifiers:
        members.setdefault(dsu.find(identifier), []).append(identifier)
    canonical = {root: "homology:" + min(member_ids) for root, member_ids in members.items()}
    group_ids = {identifier: canonical[dsu.find(identifier)] for identifier in identifiers}
    group_sizes = Counter(group_ids.values())
    size_values = sorted(group_sizes.values())
    graph_payload = {
        "alignment_method": ALIGNMENT_METHOD,
        "identity_definition": IDENTITY_DEFINITION,
        "identity_threshold": identity_threshold,
        "edges": edges,
    }
    return HomologyClusters(
        group_ids=group_ids,
        edges=edges,
        graph_sha256=hashlib.sha256(json.dumps(graph_payload, sort_keys=True).encode("utf-8")).hexdigest(),
        summary={
            "clusters": len(group_sizes), "largest_cluster": max(size_values),
            "smallest_cluster": min(size_values), "median_cluster_size": float(_median(size_values)),
            "singleton_clusters": sum(size == 1 for size in size_values),
            "clusters_larger_than_one": sum(size > 1 for size in size_values),
            "edge_count": len(edges), "biopython_version": BIOPYTHON_VERSION,
        },
    )


def _median(values: list[int]) -> float:
    middle = len(values) // 2
    return float(values[middle]) if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


@dataclass(frozen=True)
class RepresentativeClusters:
    """Deterministic V2b clusters defined only by fixed representatives."""

    group_ids: dict[str, str]
    representative_ids: tuple[str, ...]
    representative_identity: dict[str, float]
    assignment_sha256: str
    summary: dict[str, object]


def build_representative_clusters(
    records: list[dict], identity_threshold: float = 0.70
) -> RepresentativeClusters:
    """Greedily assign sequences to their closest fixed representative.

    Records are ordered by descending sequence length and then record ID.  A
    sequence joins its maximum-identity representative only when that identity
    reaches the fixed threshold; equal identities choose the lexicographically
    lowest representative ID.  Non-representative members never become bridge
    nodes, unlike V2a's single-linkage graph.
    """
    if not 0 < identity_threshold <= 1:
        raise ValueError("identity_threshold must be in (0, 1].")
    by_id: dict[str, str] = {}
    for record in records:
        identifier, sequence = str(record["record_id"]), str(record["sequence"])
        existing = by_id.get(identifier)
        if existing is not None and existing != sequence:
            raise ValueError(f"Record ID {identifier!r} has conflicting sequences.")
        by_id[identifier] = sequence
    if not by_id:
        raise ValueError("Cannot cluster an empty record collection.")
    ordered = sorted(by_id.items(), key=lambda item: (-len(item[1]), item[0]))
    representatives: list[tuple[str, str]] = []
    assigned_representative: dict[str, str] = {}
    representative_identity: dict[str, float] = {}
    for identifier, sequence in ordered:
        best_identifier: str | None = None
        best_identity = -1.0
        for representative_id, representative_sequence in representatives:
            identity = global_alignment_identity(sequence, representative_sequence)
            if identity > best_identity or (identity == best_identity and (best_identifier is None or representative_id < best_identifier)):
                best_identifier, best_identity = representative_id, identity
        if best_identifier is not None and best_identity >= identity_threshold:
            assigned_representative[identifier] = best_identifier
            representative_identity[identifier] = best_identity
        else:
            representatives.append((identifier, sequence))
            assigned_representative[identifier] = identifier
            representative_identity[identifier] = 1.0
    group_ids = {identifier: f"cluster:{representative}" for identifier, representative in assigned_representative.items()}
    size_by_representative = Counter(assigned_representative.values())
    sizes = sorted(size_by_representative.values())
    major_clusters = [
        {"cluster_id": f"cluster:{representative}", "representative_record_id": representative, "size": size}
        for representative, size in sorted(size_by_representative.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    assignment_payload = {
        "alignment_method": ALIGNMENT_METHOD,
        "identity_definition": IDENTITY_DEFINITION,
        "identity_threshold": identity_threshold,
        "assignments": sorted((identifier, assigned_representative[identifier]) for identifier in assigned_representative),
    }
    return RepresentativeClusters(
        group_ids=group_ids,
        representative_ids=tuple(representative for representative, _ in representatives),
        representative_identity=representative_identity,
        assignment_sha256=hashlib.sha256(json.dumps(assignment_payload, sort_keys=True).encode("utf-8")).hexdigest(),
        summary={
            "clusters": len(size_by_representative), "largest_cluster": max(sizes),
            "smallest_cluster": min(sizes), "median_cluster_size": float(_median(sizes)),
            "singleton_clusters": sum(size == 1 for size in sizes),
            "clusters_larger_than_one": sum(size > 1 for size in sizes),
            "cluster_size_distribution": {str(size): count for size, count in sorted(Counter(sizes).items())},
            "largest_10_clusters": major_clusters, "biopython_version": BIOPYTHON_VERSION,
        },
    )
