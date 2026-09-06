"""Installable command-line interface for the end-to-end experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .clean import clean_to_disk
from .config import load_config, load_v2_config, load_v2b_config, load_v3_config
from .fetch import fetch_to_cache, find_latest_raw, load_raw_records
from .predict import PredictionError, predict_fasta
from .train import train_experiment
from .regression import train_regression_experiment, train_representative_regression_experiment
from .v3 import train_v3_experiment


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration path (default: configs/default.yaml)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fp-predictor", description="FPbase sequence-to-brightness research baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="retrieve and cache every FPbase GraphQL page")
    fetch.add_argument("--raw-dir", default="data/raw")
    fetch.add_argument("--page-size", type=int, default=100)
    fetch.add_argument("--force", action="store_true", help="retrieve a new immutable snapshot even when cached data exist")

    clean = commands.add_parser("clean", help="validate raw records and create a processed table")
    _config_argument(clean)
    clean.add_argument("--raw", help="raw FPbase JSON; defaults to latest data/raw snapshot")
    clean.add_argument("--processed-dir", default="data/processed")

    train = commands.add_parser("train", help="grouped-CV evaluation and model serialization")
    _config_argument(train)
    train.add_argument("--data", default="data/processed/fpbase_cleaned.csv")
    train.add_argument("--results-dir", default="results")
    train.add_argument(
        "--source-metadata",
        help="raw-source metadata JSON; defaults to source_metadata.json beside --data",
    )

    train_v2 = commands.add_parser("train-v2", help="V2 homology-cluster-held-out brightness regression")
    train_v2.add_argument("--config", default="configs/v2_homology_regression.yaml")
    train_v2.add_argument("--data", default="data/processed/all_families/fpbase_cleaned.csv")
    train_v2.add_argument("--results-dir", default="results/v2_homology_regression")
    train_v2.add_argument("--source-metadata", help="raw-source metadata JSON; defaults beside --data")

    train_v2b = commands.add_parser("train-v2b", help="V2b representative-cluster-held-out brightness regression")
    train_v2b.add_argument("--config", default="configs/v2b_representative_regression.yaml")
    train_v2b.add_argument("--data", default="data/processed/all_families/fpbase_cleaned.csv")
    train_v2b.add_argument("--results-dir", default="results/v2b_representative_regression")
    train_v2b.add_argument("--source-metadata", help="raw-source metadata JSON; defaults beside --data")

    train_v3 = commands.add_parser("train-v3", help="V3 frozen-ESM2 regression on exact frozen V2b folds")
    train_v3.add_argument("--config", default="configs/v3_frozen_esm2_ridge.yaml")
    train_v3.add_argument("--data", default="data/processed/all_families/fpbase_cleaned.csv")
    train_v3.add_argument("--results-dir", default="results/v3_frozen_esm2_ridge")
    train_v3.add_argument("--cache-dir", default="data/cache/esm2_v3")
    train_v3.add_argument("--source-metadata", help="raw-source metadata JSON; defaults beside --data")

    predict = commands.add_parser("predict", help="predict tiers for one or more FASTA records")
    predict.add_argument("fasta")
    predict.add_argument("--model", required=True, help="path to results/run_.../model.joblib")
    predict.add_argument("--output", help="optional CSV output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            output = fetch_to_cache(args.raw_dir, force=args.force, page_size=args.page_size)
            print(f"Cached FPbase snapshot: {output}")
            return 0
        if args.command == "clean":
            config = load_config(args.config)
            raw_path = Path(args.raw) if args.raw else find_latest_raw("data/raw")
            records, source_metadata = load_raw_records(raw_path)
            csv_path, summary_path, summary = clean_to_disk(
                records, args.processed_dir, scope=config["scope"], state_policy=config["state_policy"],
                lineage_sidecar=config.get("lineage_sidecar"),
                near_duplicate_identity=config.get("near_duplicate_identity"),
            )
            metadata_path = Path(args.processed_dir) / "source_metadata.json"
            metadata_path.write_text(json.dumps(source_metadata, indent=2), encoding="utf-8")
            print(f"Processed dataset: {csv_path}")
            print(f"Cleaning summary: {summary_path}")
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "train":
            config = load_config(args.config)
            source_metadata = {}
            metadata_path = Path(args.source_metadata) if args.source_metadata else Path(args.data).with_name("source_metadata.json")
            if metadata_path.exists():
                source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cleaning_summary = {}
            cleaning_summary_path = Path(args.data).with_name("cleaning_summary.json")
            if cleaning_summary_path.exists():
                cleaning_summary = json.loads(cleaning_summary_path.read_text(encoding="utf-8"))
            run_directory = train_experiment(
                args.data, config, args.results_dir, source_metadata, cleaning_summary
            )
            print(f"Experiment complete: {run_directory}")
            return 0
        if args.command == "train-v2":
            config = load_v2_config(args.config)
            source_metadata = {}
            metadata_path = Path(args.source_metadata) if args.source_metadata else Path(args.data).with_name("source_metadata.json")
            if metadata_path.exists():
                source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cleaning_summary = {}
            cleaning_summary_path = Path(args.data).with_name("cleaning_summary.json")
            if cleaning_summary_path.exists():
                cleaning_summary = json.loads(cleaning_summary_path.read_text(encoding="utf-8"))
            run_directory = train_regression_experiment(
                args.data, config, args.results_dir, source_metadata, cleaning_summary
            )
            print(f"V2 experiment complete: {run_directory}")
            return 0
        if args.command == "train-v2b":
            config = load_v2b_config(args.config)
            source_metadata = {}
            metadata_path = Path(args.source_metadata) if args.source_metadata else Path(args.data).with_name("source_metadata.json")
            if metadata_path.exists():
                source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cleaning_summary = {}
            cleaning_summary_path = Path(args.data).with_name("cleaning_summary.json")
            if cleaning_summary_path.exists():
                cleaning_summary = json.loads(cleaning_summary_path.read_text(encoding="utf-8"))
            run_directory = train_representative_regression_experiment(
                args.data, config, args.results_dir, source_metadata, cleaning_summary
            )
            print(f"V2b experiment complete: {run_directory}")
            return 0
        if args.command == "train-v3":
            config = load_v3_config(args.config)
            source_metadata = {}
            metadata_path = Path(args.source_metadata) if args.source_metadata else Path(args.data).with_name("source_metadata.json")
            if metadata_path.exists():
                source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cleaning_summary = {}
            cleaning_summary_path = Path(args.data).with_name("cleaning_summary.json")
            if cleaning_summary_path.exists():
                cleaning_summary = json.loads(cleaning_summary_path.read_text(encoding="utf-8"))
            run_directory = train_v3_experiment(
                args.data, config, args.results_dir, args.cache_dir, source_metadata, cleaning_summary
            )
            print(f"V3 experiment complete: {run_directory}")
            return 0
        if args.command == "predict":
            result = predict_fasta(args.fasta, args.model)
            if args.output:
                result.to_csv(args.output, index=False)
                print(f"Predictions written to {args.output}")
            else:
                print(result.to_csv(index=False), end="")
            return 0
    except (ValueError, OSError, PredictionError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
