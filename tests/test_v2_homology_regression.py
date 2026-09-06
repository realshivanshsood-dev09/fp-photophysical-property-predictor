from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fp_predictor.config import load_v2_config, load_v2b_config
from fp_predictor.features import CompositionFeaturizer
from fp_predictor.homology import (
    build_homology_clusters,
    build_representative_clusters,
    global_alignment_identity,
    homology_edges,
)
from fp_predictor.regression import (
    inverse_transform_brightness,
    load_regression_artifact,
    make_regressor,
    regression_metrics,
    train_representative_regression_experiment,
    train_regression_experiment,
    transform_brightness,
)
from fp_predictor.split import grouped_folds


def _records():
    # A--B and B--C are 70% identical, while A--C is only 40% identical.
    return [
        {"record_id": "a", "sequence": "A" * 10, "brightness": 1.0},
        {"record_id": "b", "sequence": "A" * 7 + "C" * 3, "brightness": 50.0},
        {"record_id": "c", "sequence": "A" * 4 + "C" * 6, "brightness": 100.0},
        {"record_id": "d", "sequence": "W" * 10, "brightness": 10.0},
    ]


def _independent_frame() -> pd.DataFrame:
    residues = "ACDEFGHIKLMNPQRSTVWY"
    return pd.DataFrame([
        {"record_id": f"v2-{index}", "sequence": residue * 30, "brightness": float(index)}
        for index, residue in enumerate(residues[:10])
    ])


def test_global_alignment_identity_handles_an_unequal_length_indel():
    assert global_alignment_identity("AAAAA", "AAACAAA") == pytest.approx(5 / 7)
    clusters = build_homology_clusters([
        {"record_id": "short", "sequence": "AAAAA"},
        {"record_id": "long", "sequence": "AAACAAA"},
    ])
    assert clusters.group_ids["short"] == clusters.group_ids["long"]


def test_homology_clusters_are_transitive_deterministic_and_target_independent():
    records = _records()
    first = build_homology_clusters(records)
    second = build_homology_clusters([{**record, "brightness": 999.0} for record in records])
    assert first.group_ids == second.group_ids
    assert first.group_ids["a"] == first.group_ids["b"] == first.group_ids["c"]
    assert first.group_ids["a"] != first.group_ids["d"]
    assert ("a", "b") in first.edges and ("b", "c") in first.edges


def test_exact_duplicate_sequences_cluster_together_and_folds_ignore_brightness():
    records = [
        {"record_id": "duplicate-a", "sequence": "A" * 20, "brightness": 1.0},
        {"record_id": "duplicate-b", "sequence": "A" * 20, "brightness": 200.0},
        {"record_id": "other-a", "sequence": "C" * 20, "brightness": 2.0},
        {"record_id": "other-b", "sequence": "D" * 20, "brightness": 3.0},
    ]
    clusters = build_homology_clusters(records)
    assert clusters.group_ids["duplicate-a"] == clusters.group_ids["duplicate-b"]
    groups = np.asarray([clusters.group_ids[record["record_id"]] for record in records])
    first = [tuple(validation) for _, validation in grouped_folds(groups, 3, seed=42)]
    changed_brightness = [{**record, "brightness": 999.0} for record in records]
    changed_groups = np.asarray([
        build_homology_clusters(changed_brightness).group_ids[record["record_id"]]
        for record in changed_brightness
    ])
    second = [tuple(validation) for _, validation in grouped_folds(changed_groups, 3, seed=42)]
    assert first == second


def test_homology_edges_and_repeated_record_observations_cannot_cross_folds():
    records = _records() + [{"record_id": "a", "sequence": "A" * 10, "brightness": 200.0}]
    clusters = build_homology_clusters(records)
    groups = np.asarray([clusters.group_ids[record["record_id"]] for record in records])
    for train, validation in grouped_folds(groups, 2, seed=42):
        train_ids, validation_ids = {records[i]["record_id"] for i in train}, {records[i]["record_id"] for i in validation}
        assert not set(groups[train]).intersection(groups[validation])
        assert not ({"a", "b", "c"} & train_ids and {"a", "b", "c"} & validation_ids)
        assert not ("a" in train_ids and "a" in validation_ids)


def test_v2_transform_is_fixed_and_invertible_including_zero():
    values = np.asarray([0.0, 1.0, 9.0, 164.9])
    transformed = transform_brightness(values)
    assert transformed[0] == 0.0
    assert transformed[2] == pytest.approx(1.0)
    assert inverse_transform_brightness(transformed) == pytest.approx(values)


def test_v2_models_have_fixed_pipeline_preprocessing_and_no_target_features():
    for name in ("ridge", "elastic_net", "random_forest_regressor", "gradient_boosting_regressor"):
        model = make_regressor(name, seed=42)
        assert isinstance(model, Pipeline)
        assert isinstance(model.named_steps["features"], CompositionFeaturizer)
    names = set(CompositionFeaturizer().get_feature_names_out())
    assert not names.intersection({"brightness", "quantum_yield", "ext_coeff", "ex_max", "em_max", "pka", "lifetime", "maturation"})
    assert make_regressor("ridge", seed=42).named_steps["regressor"].alpha == 1.0
    assert make_regressor("random_forest_regressor", seed=42).named_steps["regressor"].n_estimators == 300


def test_v2_training_serializes_cluster_provenance_and_fold_assignments(tmp_path):
    data = tmp_path / "v2.csv"
    _independent_frame().to_csv(data, index=False)
    config = load_v2_config()
    run = train_regression_experiment(data, config, tmp_path / "results", {"record_count": 10})
    artifact = load_regression_artifact(run / "model.joblib")
    metadata = json.loads((run / "metadata.json").read_text())
    assignments = pd.read_csv(run / "fold_assignments.csv")
    assert artifact["target_transformation"] == "log10(brightness + 1)"
    assert metadata["homology"]["identity_threshold"] == 0.70
    assert metadata["homology"]["cluster_summary"]["clusters"] == 10
    assert len(assignments) == 10 and assignments.homology_cluster_id.nunique() == 10
    assert set(json.loads((run / "metrics.json").read_text())) == set(config["models"])


def test_v2b_representative_clusters_are_deterministic_and_target_independent():
    records = [
        {"record_id": "short", "sequence": "AAAAA", "brightness": 1.0},
        {"record_id": "long", "sequence": "AAACAAA", "brightness": 10.0},
        {"record_id": "separate", "sequence": "WWWWW", "brightness": 100.0},
    ]
    first = build_representative_clusters(records)
    second = build_representative_clusters(list(reversed([{**record, "brightness": 999.0} for record in records])))
    assert first.group_ids == second.group_ids
    assert first.group_ids["short"] == first.group_ids["long"]
    assert first.group_ids["short"] != first.group_ids["separate"]


def test_v2b_every_nonrepresentative_meets_threshold_and_representative_is_deterministic():
    records = [
        {"record_id": "long-a", "sequence": "AAAAAAB"},
        {"record_id": "long-b", "sequence": "AAAABBA"},
        {"record_id": "target-z", "sequence": "AAAAA"},
        {"record_id": "below", "sequence": "CCCCC"},
    ]
    clusters = build_representative_clusters(records)
    assert clusters.group_ids["target-z"] == "cluster:long-a"  # exact tie resolves by representative ID
    assert clusters.group_ids["below"] == "cluster:below"
    for record_id, identity in clusters.representative_identity.items():
        representative = clusters.group_ids[record_id].removeprefix("cluster:")
        if record_id == representative:
            assert identity == 1.0
        else:
            assert identity >= 0.70
    assert set(clusters.representative_ids) == {"long-a", "long-b", "below"}


def test_v2b_clusters_are_fold_isolated_and_perfect_ranking_has_rho_one():
    clusters = build_representative_clusters(_records())
    groups = np.asarray([clusters.group_ids[record["record_id"]] for record in _records()])
    for train, validation in grouped_folds(groups, 2, seed=42):
        assert not set(groups[train]).intersection(groups[validation])
    metrics = regression_metrics(np.asarray([0.0, 1.0, 2.0]), np.asarray([0.1, 1.1, 2.1]))
    assert metrics["spearman_rho"] == pytest.approx(1.0)


def test_v2b_training_serializes_representatives_and_fold_assignments(tmp_path):
    data = tmp_path / "v2b.csv"
    _independent_frame().to_csv(data, index=False)
    run = train_representative_regression_experiment(data, load_v2b_config(), tmp_path / "results", {"record_count": 10})
    metadata = json.loads((run / "metadata.json").read_text())
    assignments = pd.read_csv(run / "fold_assignments.csv")
    cluster_assignments = pd.read_csv(run / "cluster_assignments.csv")
    assert metadata["experiment"] == "v2b_representative_regression"
    assert metadata["clustering"]["clustering_strategy"] == "deterministic_representative_sequence_similarity"
    assert metadata["clustering"]["cluster_summary"]["clusters"] == 10
    assert len(assignments) == len(cluster_assignments) == 10
    assert assignments.representative_cluster_id.nunique() == 10
