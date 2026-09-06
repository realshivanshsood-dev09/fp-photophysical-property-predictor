from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from fp_predictor.config import DEFAULT_V3_AEQUOREA_DIAGNOSTIC_CONFIG, validate_v3_aequorea_diagnostic_config
from fp_predictor.clean import normalize_sequence
from fp_predictor.split import grouped_folds
from fp_predictor.v3_diagnostic import _sequence_sha256, train_v3_aequorea_diagnostic


def test_v3_aequorea_diagnostic_config_validation():
    config = dict(DEFAULT_V3_AEQUOREA_DIAGNOSTIC_CONFIG)
    assert validate_v3_aequorea_diagnostic_config(config)["target_cluster_id"] == "cluster:WQUOO"

    bad_config = dict(DEFAULT_V3_AEQUOREA_DIAGNOSTIC_CONFIG)
    bad_config["scope"] = "all_families"
    with pytest.raises(ValueError, match="Scope must be"):
        validate_v3_aequorea_diagnostic_config(bad_config)


def test_v3_aequorea_sequence_sha256_grouping():
    seq1 = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    seq2 = "mskgeelftgv  vpilveldgd vnghkfsvsgegegd atygkltlkficttgklpvpwptlvttfsygvq cfsrypdhmkqhdffksampegyvqertiffkddgnyktraevkfegdtlvnrielkgidfkedgnilghkleynynshnvyimadkqkngikvnfkirhniedgsvqladhyqqntpigdgpvllpdnhylstqsalskdpnekrdhmvllefvtaagithgmdelyk"
    assert _sequence_sha256(seq1) == _sequence_sha256(seq2)


def test_v3_aequorea_diagnostic_fold_isolation(tmp_path):
    # Verify exact sequence group disjointness across folds
    df = pd.read_csv("data/processed/all_families/fpbase_cleaned.csv")
    v2b_clusters = pd.read_csv("results/v2b_representative_regression/run_20260906T155344Z/cluster_assignments.csv")
    aeq_ids = v2b_clusters[v2b_clusters["representative_cluster_id"] == "cluster:WQUOO"]["record_id"]
    df_aeq = df[df["record_id"].isin(aeq_ids)].copy().reset_index(drop=True)
    assert len(df_aeq) == 154

    groups = df_aeq["sequence"].apply(_sequence_sha256).to_numpy()
    splits = list(grouped_folds(groups, n_splits=5, seed=42))
    assert len(splits) == 5
    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_g = set(groups[train_idx])
        val_g = set(groups[val_idx])
        assert not train_g.intersection(val_g), f"Fold {fold_num} has group leakage!"
