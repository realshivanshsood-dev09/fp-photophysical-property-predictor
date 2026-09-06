"""Explicit raw-to-processed transformations for FPbase records."""

from __future__ import annotations

import json
import math
import re
import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .lineage import build_group_ids, load_lineage_sidecar, near_duplicate_pairs

VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
# A name-only, recorded fallback because current retrieval has no FP-family
# field.  It is intentionally conservative; it is not a phylogenetic label.
GFP_NAME_PATTERN = re.compile(r"(?:^|[^a-z0-9])(?:e?gfp|av[- ]?gfp|sf[- ]?gfp|gfp)[^a-z0-9]*", re.IGNORECASE)


class CleaningError(ValueError):
    """A record cannot be safely transformed into a modelling observation."""


def normalize_sequence(value: Any) -> str:
    if not isinstance(value, str):
        raise CleaningError("sequence is missing or not text")
    sequence = "".join(value.split()).upper()
    if not sequence:
        raise CleaningError("sequence is empty")
    invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
    if invalid:
        raise CleaningError("sequence contains unsupported amino acids: " + ", ".join(invalid))
    return sequence


def _state_values(record: dict[str, Any], policy: str) -> list[tuple[dict[str, Any], str]]:
    if policy == "default_only":
        state = record.get("defaultState")
        if state is None:  # REST-compatible small fixture spelling
            state = record.get("default_state")
        if not isinstance(state, dict):
            return []
        state_id = str(state.get("id") or state.get("slug") or "default")
        return [(state, state_id)]
    states = record.get("states") or []
    return [(state, str(state.get("id") or state.get("slug") or index)) for index, state in enumerate(states) if isinstance(state, dict)]


def _field(state: dict[str, Any], camel: str, snake: str) -> Any:
    return state.get(camel, state.get(snake))


def _gfp_name_match(record: dict[str, Any]) -> bool:
    labels = [record.get("name", ""), record.get("slug", ""), record.get("baseName", "")]
    labels.extend(record.get("aliases") or [])
    return any(isinstance(label, str) and GFP_NAME_PATTERN.search(label) for label in labels)


def _provenance_value(record: dict[str, Any], key: str) -> Any:
    value = record.get(key)
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def clean_records(
    raw_records: Iterable[dict[str, Any]],
    *,
    scope: str = "gfp_only",
    state_policy: str = "default_only",
    lineage_sidecar: str | Path | None = None,
    near_duplicate_identity: float | None = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate records, retain continuous brightness, and assign leakage groups.

    ``all_states`` makes protein-state rows the unit of analysis; its state IDs
    are retained and all states from a protein remain in one group.
    """
    raw = list(raw_records)
    summary: dict[str, Any] = {
        "raw_proteins": len(raw), "usable_sequences": 0, "proteins_with_brightness": 0,
        "removed_missing_sequence": 0, "removed_invalid_sequence": 0,
        "removed_missing_brightness": 0, "removed_invalid_values": 0,
        "excluded_scope": 0, "excluded_states": 0, "duplicates": 0,
        "sequence_duplicate_records": 0,
        "exact_duplicate_sequence_groups": 0,
        "conflicting_brightness_exact_sequence_groups": 0,
        "state_policy": state_policy, "scope": scope,
        "scope_eligible_sequences": 0, "scope_eligible_records_with_state": 0,
        "all_valid_sequence_records_with_state": 0,
    }
    observations: list[dict[str, Any]] = []
    seen_observations: set[tuple[str, str]] = set()
    for raw_record in raw:
        if not isinstance(raw_record, dict):
            summary["removed_invalid_values"] += 1
            continue
        identifier = raw_record.get("id") or raw_record.get("uuid")
        if identifier is None:
            summary["removed_invalid_values"] += 1
            continue
        try:
            sequence = normalize_sequence(raw_record.get("seq", raw_record.get("sequence")))
        except CleaningError as error:
            if "missing" in str(error) or "empty" in str(error):
                summary["removed_missing_sequence"] += 1
            else:
                summary["removed_invalid_sequence"] += 1
            continue
        summary["usable_sequences"] += 1
        states = _state_values(raw_record, state_policy)
        if states:
            summary["all_valid_sequence_records_with_state"] += 1
        if scope == "gfp_only" and not _gfp_name_match(raw_record):
            summary["excluded_scope"] += 1
            continue
        summary["scope_eligible_sequences"] += 1
        if not states:
            summary["excluded_states"] += 1
            continue
        summary["scope_eligible_records_with_state"] += 1
        for state, state_id in states:
            brightness = _field(state, "brightness", "brightness")
            try:
                brightness = float(brightness)
            except (TypeError, ValueError):
                summary["removed_missing_brightness"] += 1
                continue
            if not math.isfinite(brightness) or brightness < 0:
                summary["removed_invalid_values"] += 1
                continue
            observation_key = (str(identifier), str(state_id))
            if observation_key in seen_observations:
                summary["duplicates"] += 1
                continue
            seen_observations.add(observation_key)
            summary["proteins_with_brightness"] += 1
            observations.append({
                "record_id": str(identifier), "state_id": str(state_id),
                "name": raw_record.get("name", ""), "slug": raw_record.get("slug", ""),
                "base_name": raw_record.get("baseName", raw_record.get("base_name", "")),
                "sequence": sequence, "brightness": brightness,
                "sequence_source_validated": raw_record.get("seqValidated", raw_record.get("seq_validated")),
                "cofactor": raw_record.get("cofactor"),
                "switch_type": raw_record.get("switchType", raw_record.get("switch_type")),
                "chromophore": raw_record.get("chromophore"),
                "ext_coeff": _field(state, "extCoeff", "ext_coeff"),
                "quantum_yield": _field(state, "qy", "qy"),
                "ex_max": _field(state, "exMax", "ex_max"),
                "em_max": _field(state, "emMax", "em_max"),
                "lifetime": _field(state, "lifetime", "lifetime"),
                "maturation": _field(state, "maturation", "maturation"),
                "pka": _field(state, "pka", "pka"),
                "is_dark": _field(state, "isDark", "is_dark"),
                "state_source_map": _provenance_value(state, "sourceMap"),
                "parent_organism": _provenance_value(raw_record, "parentOrganism"),
                "primary_reference": _provenance_value(raw_record, "primaryReference"),
                "references": _provenance_value(raw_record, "references"),
            })
    if not observations:
        raise CleaningError("No usable observations remain after cleaning; inspect the summary and scope.")
    parents = load_lineage_sidecar(lineage_sidecar) if lineage_sidecar else None
    if lineage_sidecar:
        sidecar_path = Path(lineage_sidecar)
        summary["lineage_sidecar"] = {
            "path": str(sidecar_path),
            "sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        }
    else:
        summary["lineage_sidecar"] = None
    sequence_counts: dict[str, int] = {}
    for observation in observations:
        sequence_counts[observation["sequence"]] = sequence_counts.get(observation["sequence"], 0) + 1
    summary["sequence_duplicate_records"] = sum(count - 1 for count in sequence_counts.values() if count > 1)
    duplicate_sequences = {sequence: count for sequence, count in sequence_counts.items() if count > 1}
    summary["exact_duplicate_sequence_groups"] = len(duplicate_sequences)
    summary["conflicting_brightness_exact_sequence_groups"] = sum(
        len({row["brightness"] for row in observations if row["sequence"] == sequence}) > 1
        for sequence in duplicate_sequences
    )
    near_pairs = near_duplicate_pairs(observations, near_duplicate_identity)
    summary["near_duplicate_identity"] = near_duplicate_identity
    summary["same_length_near_duplicate_pairs"] = len(near_pairs)
    summary["records_in_near_duplicate_pairs"] = len({identifier for pair in near_pairs for identifier in pair})
    group_ids = build_group_ids(
        observations, parents=parents, near_duplicate_identity=near_duplicate_identity
    )
    for row in observations:
        row["group_id"] = group_ids[row["record_id"]]
        if parents:
            row["grouping_source"] = "verified_lineage_plus_sequence_neighbour"
        elif near_duplicate_identity is not None:
            row["grouping_source"] = "exact_plus_same_length_sequence_neighbour"
        else:
            row["grouping_source"] = "exact_sequence_only"
        try:
            ec, qy = float(row["ext_coeff"]), float(row["quantum_yield"])
            # FPbase brightness is conventionally EC × QY / 1000.  Preserve
            # its supplied brightness as target and retain reconstruction only
            # as a scale-aware QA diagnostic.
            reconstructed = ec * qy / 1000 if math.isfinite(ec) and math.isfinite(qy) else None
            row["brightness_from_ec_qy"] = reconstructed
            row["brightness_ec_qy_difference"] = row["brightness"] - reconstructed if reconstructed is not None else None
        except (TypeError, ValueError):
            row["brightness_from_ec_qy"] = None
            row["brightness_ec_qy_difference"] = None
    frame = pd.DataFrame(observations).sort_values(["record_id", "state_id"]).reset_index(drop=True)
    brightness = frame["brightness"].astype(float)
    summary["brightness_distribution"] = {
        "min": float(brightness.min()), "max": float(brightness.max()),
        "median": float(brightness.median()), "mean": float(brightness.mean()),
        "q1": float(brightness.quantile(0.25)), "q3": float(brightness.quantile(0.75)),
    }
    diagnostic = frame["brightness_ec_qy_difference"].dropna().astype(float)
    summary["brightness_ec_qy_qa"] = {
        "available_records": int(len(diagnostic)),
        "mean_absolute_difference": float(diagnostic.abs().mean()) if len(diagnostic) else None,
        "max_absolute_difference": float(diagnostic.abs().max()) if len(diagnostic) else None,
    }
    summary["attrition"] = {
        "raw_records": len(raw),
        "valid_sequence_records": summary["usable_sequences"],
        "valid_sequence_records_with_usable_state_all_scopes": summary["all_valid_sequence_records_with_state"],
        "scope_eligible_valid_sequence_records": summary["scope_eligible_sequences"],
        "scope_eligible_records_with_usable_state": summary["scope_eligible_records_with_state"],
        "scope_eligible_brightness_observations": summary["proteins_with_brightness"],
        "final_numerically_valid_observations": len(frame),
    }
    summary["final_dataset"] = len(frame)
    summary["unique_groups"] = int(frame["group_id"].nunique())
    if parents:
        summary["grouping_source"] = "verified_lineage_plus_sequence_neighbour"
    elif near_duplicate_identity is not None:
        summary["grouping_source"] = "exact_plus_same_length_sequence_neighbour"
    else:
        summary["grouping_source"] = "exact_sequence_only"
    summary["scope_method"] = "name_heuristic_fallback" if scope == "gfp_only" else "no_family_filter"
    return frame, summary


def clean_to_disk(
    raw_records: Iterable[dict[str, Any]],
    output_dir: str | Path = "data/processed",
    **kwargs: Any,
) -> tuple[Path, Path, dict[str, Any]]:
    frame, summary = clean_records(raw_records, **kwargs)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "fpbase_cleaned.csv"
    summary_path = destination / "cleaning_summary.json"
    frame.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path, summary
