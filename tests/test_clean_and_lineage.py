from __future__ import annotations

import pytest

from fp_predictor.clean import clean_records, normalize_sequence
from fp_predictor.lineage import build_group_ids, lineage_roots
from tests.conftest import fixture_records


def test_sequence_validation_rejects_noncanonical_residues():
    with pytest.raises(ValueError, match="unsupported"):
        normalize_sequence("MAZ")


def test_clean_retains_continuous_target_and_groups_exact_duplicates():
    records = fixture_records()
    records.extend([
        {"id": "p6", "name": "GFP-duplicate-sequence", "seq": records[0]["seq"], "defaultState": {"id": "s6", "brightness": 7.0}},
        {"id": "p7", "name": "GFP-missing-sequence", "defaultState": {"id": "s7", "brightness": 7.0}},
        {"id": "p8", "name": "GFP-invalid-value", "seq": records[0]["seq"], "defaultState": {"id": "s8", "brightness": -1.0}},
    ])
    frame, summary = clean_records(records, scope="all_families", near_duplicate_identity=None)
    assert set(frame["record_id"]) == {"p1", "p2", "p5", "p6"}
    assert frame["brightness"].dtype.kind == "f"
    assert frame.loc[frame.record_id == "p1", "group_id"].iloc[0] != frame.loc[frame.record_id == "p2", "group_id"].iloc[0]
    assert frame.loc[frame.record_id == "p1", "group_id"].iloc[0] == frame.loc[frame.record_id == "p6", "group_id"].iloc[0]
    assert summary["removed_missing_brightness"] == 1
    assert summary["removed_invalid_sequence"] == 1
    assert summary["removed_missing_sequence"] == 1
    assert summary["removed_invalid_values"] == 1
    reconstructed = frame.loc[frame.record_id == "p1", "brightness_from_ec_qy"].iloc[0]
    assert reconstructed == pytest.approx(5.0)


def test_all_states_is_explicitly_state_level_but_keeps_protein_grouped():
    frame, _ = clean_records(fixture_records(), scope="all_families", state_policy="all_states")
    p5 = frame[frame.record_id == "p5"]
    assert len(p5) == 2
    assert p5.group_id.nunique() == 1


def test_same_length_near_duplicates_are_grouped_without_claiming_lineage():
    records = [
        {"id": "first", "name": "first", "seq": "M" + "A" * 99, "defaultState": {"id": "a", "brightness": 1}},
        {"id": "second", "name": "second", "seq": "M" + "A" * 98 + "C", "defaultState": {"id": "b", "brightness": 2}},
        {"id": "third", "name": "third", "seq": "M" + "A" * 100, "defaultState": {"id": "c", "brightness": 3}},
    ]
    frame, summary = clean_records(records, scope="all_families", near_duplicate_identity=0.95)
    assert frame.loc[frame.record_id == "first", "group_id"].iloc[0] == frame.loc[frame.record_id == "second", "group_id"].iloc[0]
    assert frame.loc[frame.record_id == "first", "group_id"].iloc[0] != frame.loc[frame.record_id == "third", "group_id"].iloc[0]
    assert summary["same_length_near_duplicate_pairs"] == 1


def test_near_duplicate_components_are_transitive_and_deterministic():
    # A--B and B--C satisfy 95%; A--C does not.  Connected components are the
    # intended leakage-control unit, independent of the brightness values.
    records = [
        {"record_id": "a", "sequence": "A" * 100},
        {"record_id": "b", "sequence": "C" * 5 + "A" * 95},
        {"record_id": "c", "sequence": "C" * 10 + "A" * 90},
        {"record_id": "below_threshold", "sequence": "C" * 20 + "A" * 80},
        {"record_id": "different_length", "sequence": "A" * 101},
    ]
    first = build_group_ids(records, near_duplicate_identity=0.95)
    second = build_group_ids(records, near_duplicate_identity=0.95)
    assert first == second
    assert first["a"] == first["b"] == first["c"]
    assert first["a"] != first["below_threshold"]
    assert first["a"] != first["different_length"]


def test_grouping_is_independent_of_target_values():
    records = [
        {"record_id": "a", "sequence": "A" * 100, "brightness": 0.01},
        {"record_id": "b", "sequence": "C" + "A" * 99, "brightness": 164.9},
    ]
    changed_targets = [{**record, "brightness": 42.0} for record in records]
    assert build_group_ids(records, near_duplicate_identity=0.95) == build_group_ids(
        changed_targets, near_duplicate_identity=0.95
    )


def test_default_state_preserves_the_source_state_identifier():
    frame, _ = clean_records(fixture_records(), scope="all_families")
    assert frame.loc[frame.record_id == "p1", "state_id"].iloc[0] == "s1"


def test_lineage_handles_chains_missing_parents_and_cycles():
    roots = lineage_roots(["a", "b", "c", "d", "e"], {"b": "a", "c": "b", "d": "gone", "e": "e"})
    assert roots["a"] == roots["b"] == roots["c"] == "a"
    assert roots["d"] == "missing-parent:gone"
    assert roots["e"] == "cycle:e"


def test_verified_lineage_and_sequence_identity_are_unioned():
    records = [
        {"record_id": "a", "sequence": "MA"}, {"record_id": "b", "sequence": "MC"},
        {"record_id": "c", "sequence": "MA"},
    ]
    groups = build_group_ids(records, {"b": "a"})
    assert groups["a"] == groups["b"] == groups["c"]
