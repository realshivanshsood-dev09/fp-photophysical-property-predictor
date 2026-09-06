from __future__ import annotations

import json

import pandas as pd
import pytest

from fp_predictor.cli import main
from fp_predictor.config import load_config
from fp_predictor.predict import PredictionError, predict_fasta
from fp_predictor.train import derive_thresholds, select_model, train_experiment
from tests.conftest import training_frame


def test_training_serialization_and_multifasta_prediction(tmp_path):
    data = tmp_path / "clean.csv"
    training_frame().to_csv(data, index=False)
    config = load_config()
    config["scope"] = "all_families"
    config["near_duplicate_identity"] = None
    run = train_experiment(data, config, tmp_path / "results", {"record_count": 15})
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics) == set(config["models"])
    assert (run / "fold_assignments.csv").exists()
    assert (run / "cleaning_summary.json").exists()
    summary = json.loads((run / "dataset_summary.json").read_text())
    assert summary["feature_count"] == 39
    assignments = pd.read_csv(run / "fold_assignments.csv")
    fold_one_validation = set(assignments.loc[assignments.fold == 1, "record_id"])
    expected = derive_thresholds(
        training_frame().loc[~training_frame().record_id.isin(fold_one_validation), "brightness"]
    )
    observed = metrics["logistic_regression"]["folds"][0]["thresholds"]
    assert observed == pytest.approx(expected)
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">first\nMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGCD\n>second\nMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGEF\n")
    result = predict_fasta(fasta, run / "model.joblib")
    assert list(result["protein"]) == ["first", "second"]
    assert set(result["predicted_brightness_tier"]).issubset({"LOW", "MEDIUM", "HIGH"})
    assert result.filter(like="probability_").shape == (2, 3)


def test_invalid_fasta_fails_cleanly(tmp_path):
    fasta = tmp_path / "bad.fasta"
    fasta.write_text("MAZ\n")
    with pytest.raises(PredictionError, match="precedes"):
        predict_fasta(fasta, tmp_path / "missing.joblib")


def test_cli_predict_writes_csv(tmp_path):
    data = tmp_path / "clean.csv"
    training_frame().to_csv(data, index=False)
    config = load_config()
    config["scope"] = "all_families"
    config["near_duplicate_identity"] = None
    run = train_experiment(data, config, tmp_path / "results")
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">one\nMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGGGGGGGGGGGGGGGGGGGGCD\n")
    output = tmp_path / "predictions.csv"
    assert main(["predict", str(fasta), "--model", str(run / "model.joblib"), "--output", str(output)]) == 0
    assert output.exists()


def test_model_selection_uses_fold_mean_and_simplicity_rule():
    models = {
        "majority": {"aggregate": {"macro_f1": {"mean": 0.90, "std": 0.05}}},
        "logistic_regression": {"aggregate": {"macro_f1": {"mean": 0.50, "std": 0.04}}},
        "random_forest": {"aggregate": {"macro_f1": {"mean": 0.52, "std": 0.04}}},
        "gradient_boosting": {"aggregate": {"macro_f1": {"mean": 0.40, "std": 0.02}}},
    }
    selected, explanation = select_model(models)
    # LR is within the RF top model's 0.04 SD and wins the intended simplicity tie-break.
    assert selected == "logistic_regression"
    assert explanation["policy"] == "one_standard_deviation_simplicity"
