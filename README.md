# FP Photophysical Property Predictor

A small, reproducible fluorescent-protein (FP) sequence-to-photophysics framework. V1 asks a deliberately limited question: can compact, sequence-derived physicochemical descriptors predict **FPbase brightness** tiers when exact and obviously near-identical sequences are kept out of the same validation fold?

This is research software and an exploratory benchmark—not a validated FP design tool or a claim of biological mechanism.

## Current validated snapshot

The checked run uses an immutable public FPbase GraphQL snapshot retrieved on 2026-09-06. It contains 1,042 proteins. Under `default_only` state handling:

| Stage | All families | GFP-name heuristic |
| --- | ---: | ---: |
| Raw records | 1,042 | 1,042 |
| Canonical-sequence records | 977 | 977 |
| Records with a usable state | 856 | 59 after scope filter |
| FPbase brightness available | 601 | 30 |
| Numerically valid observations | 601 | 30 |
| Sequence-neighborhood groups (95%) | 215 | 9 |

The broad cohort excludes 51 missing sequences, 14 invalid sequences, 121 records without an eligible default state, and 255 records lacking brightness. The small GFP subset is primarily caused by the **name/alias heuristic**, which leaves only 70 candidate sequences; it is not an API retrieval failure. Complete machine-readable attrition reports are saved with every run.

The default primary scope is therefore `all_families` (601 observations), with the GFP-only configuration retained as a sensitivity analysis. No authoritative FP family label was exposed by the audited public API, so `gfp_only` is explicitly exploratory and should not be read as a verified GFP-family cohort.

## Data source and state policy

Acquisition uses FPbase’s supported public GraphQL entrypoint, `https://www.fpbase.org/graphql/`, via cursor-paginated `allProteins`. The raw response, source endpoint, retrieval time, query hash, content hash, and record count are saved under `data/raw/`; raw snapshots are ignored by Git.

Each FPbase protein can have several states. `state_policy: default_only` uses FPbase’s API-designated `defaultState` and preserves its original state ID. `all_states` is supported, but makes protein-state rows the analysis unit; all states from one protein remain in the same leakage-control group.

FPbase brightness is the continuous canonical target throughout cleaning. When both measurements are present, `extCoeff × qy / 1000` is recorded only as a QA reconstruction. In the validated all-family run it was available for 600 records, with mean absolute difference 0.00099 brightness units and maximum 0.005 from supplied FPbase brightness. It is never a model feature or a replacement target.

The all-family set includes 550 records with no listed cofactor and 51 with a listed cofactor (mostly biliverdin or flavin). Cofactor is retained as provenance, not a feature or silent exclusion. This broadens coverage but is a real limitation: a sequence-only model cannot represent cofactor availability or experimental context.

## Leakage control: sequence grouping, not lineage grouping

FPbase documentation describes protein lineages, but its audited public GraphQL types do **not** expose the parent/child edges required to recover engineering ancestry. `parent_organism` is retained as metadata only and is never used as a lineage proxy. Consequently, this project does not claim lineage-aware evaluation from public FPbase data alone.

Default leakage-control groups are connected components of:

1. exact amino-acid sequence identity; and
2. same-length sequence pairs with positional identity at least 95%.

The second rule is configurable through `near_duplicate_identity` and is independent of brightness labels. Connected components are intentional: if A is sufficiently similar to B and B to C, all three are held together even when A and C do not directly meet the threshold. This conservative point-mutant safeguard found 3,120 pairs involving 465 of 601 observations in the current all-family snapshot. It catches same-length variant series, but is **not** a substitute for ancestry or comprehensive homology control: it performs no alignment, does not group unequal-length homologues, and does not make the 215 groups biologically independent lineages.

An optional verified `child_id,parent_id` CSV sidecar can merge explicit ancestry groups with sequence groups. It rejects multiple parents, handles missing ancestors deterministically, and detects cycles. Without a sidecar, the correct description is *sequence-neighborhood grouped CV*.

## Features and target-leakage boundary

The model matrix has 39 deterministic sequence-only features (feature version 1.1):

- 20 amino-acid frequencies and length;
- acidic, basic, charged, polar, nonpolar, hydrophobic, and aromatic fractions;
- net-charge proxy at pH 7 and a Henderson–Hasselbalch pI estimate;
- mean and standard deviation of Kyte–Doolittle hydropathy;
- glycine, proline, and cysteine fractions;
- aliphatic index, molecular-weight estimate, Shannon sequence entropy, and residue diversity.

Brightness, EC, QY, EC×QY, spectral maxima, lifetime, maturation, pKa, cofactor, source maps, references, and all other FPbase measurements are excluded from model inputs. Global features do not capture the three-dimensional chromophore environment. No chromophore-local feature is included: there is no robust sequence-position mapping valid across the broad family scope.

## Evaluation protocol and results

`LOW`, `MEDIUM`, and `HIGH` are operational training-distribution tertiles, not biological brightness classes. Brightness remains continuous; classification imposes dataset-derived boundaries on a noisy continuous measurement and can create boundary effects. In every fold, the two thresholds are fit only to that fold’s training brightness values and then applied to validation records. Final deployment thresholds are fit once on the designated complete training data and saved in the artifact. Regression or ranking on continuous brightness is a v2 direction, not part of this benchmark.

Folds allocate whole sequence-neighborhood groups with a seeded, sample-count-balanced `sequence_neighborhood_group_kfold` algorithm; brightness is not used to construct folds. The code asserts no train/validation group overlap. Scaling is inside the logistic-regression pipeline. Majority and stratified-random baselines use training labels only.

The validated all-family 5-fold run (`601` observations; `215` groups; class counts `199/201/201`) produced:

| Model | Macro-F1 (mean ± SD) | Balanced accuracy | Accuracy |
| --- | --- | --- | --- |
| Majority | 0.170 ± 0.028 | 0.333 ± 0.000 | 0.346 ± 0.072 |
| Stratified random | 0.313 ± 0.044 | 0.318 ± 0.046 | 0.318 ± 0.042 |
| Logistic regression | 0.440 ± 0.036 | 0.453 ± 0.034 | 0.476 ± 0.044 |
| Random forest | 0.520 ± 0.045 | 0.525 ± 0.046 | 0.551 ± 0.022 |
| Gradient boosting | 0.454 ± 0.048 | 0.466 ± 0.045 | 0.479 ± 0.058 |

Random forest is selected for that run: it is the only learned model within one top-model standard deviation under the stored model-selection policy. Per-fold precision, recall, F1, supports, aggregate per-class summaries, and confusion matrices are saved in the run directory. The result is exploratory evidence of signal under this particular sequence-neighborhood grouped CV—not evidence of performance on homology-separated or novel FP scaffolds, universal mechanistic prediction across FP families, or prospective FP-design utility.

The validated artifacts are [`results/all_families/run_20260906T150216Z`](results/all_families/run_20260906T150216Z) and the GFP sensitivity run [`results/gfp_only/run_20260906T150237Z`](results/gfp_only/run_20260906T150237Z).

The GFP-name sensitivity analysis has 30 observations but only 9 sequence-neighborhood groups. It uses 3 grouped folds because 5 folds would produce 3–10-observation validation folds with absent classes. Its results are too sparse for meaningful model selection and should not be compared directly with the all-family experiment:

| Model | Macro-F1 (mean ± SD) | Balanced accuracy | Accuracy |
| --- | --- | --- | --- |
| Majority | 0.194 ± 0.057 | 0.444 ± 0.079 | 0.433 ± 0.189 |
| Stratified random | 0.280 ± 0.013 | 0.397 ± 0.068 | 0.367 ± 0.047 |
| Logistic regression | 0.192 ± 0.053 | 0.390 ± 0.131 | 0.333 ± 0.094 |
| Random forest | 0.209 ± 0.105 | 0.327 ± 0.237 | 0.333 ± 0.170 |
| Gradient boosting | 0.291 ± 0.079 | 0.453 ± 0.187 | 0.467 ± 0.125 |

`predict_proba` output is an uncalibrated model class probability (for random forest, a tree-vote fraction), not calibrated confidence; it has not undergone held-out calibration validation.

## Installation and use

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -e .[dev]

# Immutable current-data snapshot
fp-predictor fetch --force

# Primary all-family workflow
fp-predictor clean --config configs/default.yaml
fp-predictor train --config configs/default.yaml

# Use the exact run printed by train
fp-predictor predict examples/example.fasta --model results/run_.../model.joblib

# Exploratory GFP-name sensitivity analysis
fp-predictor clean --config configs/gfp_only.yaml --processed-dir data/processed/gfp_only
fp-predictor train --config configs/gfp_only.yaml --data data/processed/gfp_only/fpbase_cleaned.csv --results-dir results/gfp_only
```

The predict command accepts multi-FASTA input and can write a CSV via `--output`. It rejects malformed FASTA, empty sequences, unsupported residues, missing models, and incompatible artifacts.

Each training run writes `model.joblib`, `metadata.json`, `metrics.json`, `confusion_matrices.json`, `fold_assignments.csv`, `cleaning_summary.json`, and `dataset_summary.json`. The artifact includes preprocessing, classifier, class mapping, thresholds, and feature version.

## Validation and prior art

```bash
pytest
```

Tests use local fixtures and cover GraphQL pagination/malformed responses, cleaning attrition, state IDs, EC×QY QA, exact and near-duplicate grouping, transitive components, sidecar chains/missing parents/cycles, deterministic features, leakage-free groups, training-only thresholds, serialization, and FASTA prediction.

FPredX is Tam C, Zhang KYJ, *FPredX: Interpretable models for the prediction of spectral maxima, brightness, and oligomeric states of fluorescent proteins*, **Proteins** 90(3):732–746 (2022; online 2021), [doi:10.1002/prot.26270](https://doi.org/10.1002/prot.26270). It uses aligned one-hot sequences and gradient-boosted trees for several FP targets. Its data, splits, and evaluation differ from this repository, so no direct numerical comparison is claimed. The cited “Wu et al., Proteins, 2024” attribution is not the verified FPredX publication. No unambiguous primary source was found for the requested 2026 FPbase QY/structure-graph work or a specific RFP-variant claim, so neither is cited as established prior art here.

## Limitations and next step

FPbase aggregates measurements from heterogeneous experimental contexts; cofactors, pH, maturation, oligomerization, chromophore chemistry, and structural architecture are not represented in a sequence-only global input. The all-family primary cohort deliberately pools proteins that can differ in these mechanisms, so this experiment does not establish universal mechanistic prediction across FP families. Similarity grouping leaves unresolved ancestry, broader homology, and indel-related leakage. The GFP cohort is a small, heuristic sensitivity analysis and is underpowered for GFP-specific conclusions. V1 is engineering-complete and scientifically honest as a sequence-neighborhood-controlled exploratory baseline. The highest-value next step is a verified FPbase lineage/family export, followed by a stronger homology-aware evaluation—not a larger model or a structure-prediction system.

## V2a: alignment-based single-linkage regression sensitivity analysis

V2a and V2b are separate continuous-brightness experiments on the same 601-observation all-family, default-state dataset. They preserve V1 unchanged. The target is transformed deterministically as `log10(brightness + 1)` (including zero brightness); the inverse is saved with the artifact. This is an operational statistical transformation, not a physical law of fluorescence.

V2 uses the same corrected 39 sequence-only features (v1.1). Sequence length remains an explicit descriptor and can act as a broad architectural/family proxy. No FPbase photophysical measurement enters the model matrix.

For V2a and V2b, a 70% alignment-based sequence-identity threshold was pre-specified as an operational homology-clustering criterion to provide a substantially stricter generalization test than V1 near-duplicate grouping. It is **not** a universal definition of biological homology. Pairs are aligned using global Needleman--Wunsch alignment with match `+1`, mismatch `-1`, and linear gap penalty `-1` per residue. Identity is exactly `identical residue-residue alignment columns / all global-alignment columns, including gap columns`.

In V2a, an edge joins records at identity >= 0.70; deterministic connected components are held out as whole units. Thus unequal-length sequences are eligible to cluster and A--B--C chains are held together transitively. This is *alignment-based single-linkage sequence-similarity clustering*, not lineage-aware CV: public FPbase GraphQL still lacks engineering parent/child edges.

The fresh V2a run produced 64 clusters (23 singleton clusters; 41 clusters with >1 observation; median size 2; largest size 154; 20,244 graph edges), so the pre-specified five folds are feasible but uneven cluster sizes remain an important uncertainty. Fold allocation uses only cluster IDs, their sizes, and seed 42.

Both V2 experiments evaluate mean and median baselines plus fixed-hyperparameter Ridge, Elastic Net, random-forest regressor, and gradient-boosting regressor. The pre-specified selection criterion is mean validation Spearman rho; Pearson r, RMSE, MAE, and R² are computed on `log10(brightness + 1)`. Spearman is unchanged by this monotonic transform and is rank-equivalent to raw brightness. Raw-space RMSE/MAE are retained only as supplementary artifacts. Constant baselines have undefined Spearman rho.

| Model | Spearman rho | Pearson r | RMSE (log) | MAE (log) | R² (log) |
| --- | --- | --- | --- | --- | --- |
| Mean baseline | undefined | -0.000 ± 0.000 | 0.502 ± 0.081 | 0.412 ± 0.069 | -0.088 ± 0.110 |
| Median baseline | undefined | -0.000 ± 0.000 | 0.508 ± 0.095 | 0.408 ± 0.082 | -0.099 ± 0.071 |
| Ridge | -0.024 ± 0.090 | -0.045 ± 0.126 | 0.604 ± 0.053 | 0.474 ± 0.048 | -0.628 ± 0.348 |
| Elastic Net | -0.026 ± 0.072 | -0.040 ± 0.118 | 0.563 ± 0.054 | 0.443 ± 0.048 | -0.411 ± 0.295 |
| Random forest regressor | -0.011 ± 0.080 | 0.067 ± 0.144 | 0.523 ± 0.088 | 0.419 ± 0.064 | -0.175 ± 0.066 |
| Gradient boosting regressor | -0.008 ± 0.120 | 0.060 ± 0.131 | 0.516 ± 0.085 | 0.417 ± 0.067 | -0.142 ± 0.071 |

Gradient boosting is selected mechanically by the prespecified mean-Spearman rule, not because this establishes useful prediction. Under this strict V2a cluster definition, the result does not demonstrate above-baseline rank prediction; it does not establish mechanistic understanding or prospective design utility.

Run V2 after the standard fetch/clean commands:

```bash
fp-predictor train-v2 --config configs/v2_homology_regression.yaml \
  --data data/processed/all_families/fpbase_cleaned.csv \
  --results-dir results/v2_homology_regression
```

The validated V2a artifacts are [`results/v2_homology_regression/run_20260906T152552Z`](results/v2_homology_regression/run_20260906T152552Z). They include the model, alignment-graph edge list/hash, cluster statistics, fold assignments, fold metrics, fixed hyperparameters, source provenance, and software versions.

## V2b: alignment-based representative clustering

V2b is the primary V2 evaluation. It uses the exact same dataset, target transformation, 39 v1.1 sequence-only features, aligner, scoring, identity denominator, 70% operational threshold, seed, folds, models, and metrics as V2a. The sole methodological difference is clustering.

Sequences are sorted by descending length and then record ID. Each is compared with every existing cluster representative and assigned to the maximum-identity representative when that identity is at least 70%; exact ties use the lowest representative record ID. Otherwise, it becomes a new representative. Cluster IDs are `cluster:<representative_record_id>`. Non-representative members never act as bridges, so this deterministic representative-based sequence-similarity clustering is not V2a single linkage. It is neither true engineering lineage, phylogenetic clade assignment, nor a definitive biological-family definition.

V2b produced 74 clusters: 25 singleton clusters, 49 multi-observation clusters, median size 2.5, and largest size 154. Its ten largest representatives are WQUOO (154), 45EFN (75), 1LT8G (65), 6LRFZ (46), AJCVH (27), 9JNRR (12), 2EG15 (11), T7XX8 (11), GNU4X (10), and BUJNN (9). The remaining 154-member cluster still makes one five-fold validation partition especially stringent and contributes substantial fold uncertainty.

| Model | Spearman rho | Pearson r | RMSE (log) | MAE (log) | R² (log) |
| --- | --- | --- | --- | --- | --- |
| Mean baseline | undefined | 0.000 ± 0.000 | 0.501 ± 0.056 | 0.407 ± 0.043 | -0.015 ± 0.014 |
| Median baseline | undefined | 0.000 ± 0.000 | 0.507 ± 0.063 | 0.402 ± 0.048 | -0.037 ± 0.039 |
| Ridge | 0.042 ± 0.293 | 0.099 ± 0.229 | 0.563 ± 0.076 | 0.438 ± 0.055 | -0.320 ± 0.333 |
| Elastic Net | 0.039 ± 0.299 | 0.092 ± 0.239 | 0.539 ± 0.070 | 0.419 ± 0.055 | -0.200 ± 0.253 |
| Random forest regressor | 0.176 ± 0.263 | 0.214 ± 0.246 | 0.492 ± 0.060 | 0.390 ± 0.041 | -0.001 ± 0.207 |
| Gradient boosting regressor | 0.221 ± 0.247 | 0.249 ± 0.239 | 0.479 ± 0.062 | 0.379 ± 0.043 | 0.063 ± 0.154 |

Gradient boosting is selected by the same pre-specified mean-Spearman rule. V2b’s positive mean association and lower mean errors versus the location baselines are exploratory evidence under this alternative partition, but its large fold variability means it does not establish robust homology-independent prediction, mechanistic causality, or prospective FP-design performance.

```bash
fp-predictor train-v2b --config configs/v2b_representative_regression.yaml \
  --data data/processed/all_families/fpbase_cleaned.csv \
  --results-dir results/v2b_representative_regression
```

The validated V2b artifacts are [`results/v2b_representative_regression/run_20260906T155344Z`](results/v2b_representative_regression/run_20260906T155344Z). They include model and source metadata, fixed alignment parameters, assignment hash, cluster statistics, representative identities, observation-to-cluster and observation-to-fold maps, and per-fold/aggregate metrics.
