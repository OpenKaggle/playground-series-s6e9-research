# Playground Series S6E9 — research archive

An OpenKaggle source-and-provenance archive for the archived `Will_Buy_EV`
ROC-AUC campaign. This is not a copy of the competition workspace.

The repository contains user-authored training/validation code, compact
aggregate experiment records, and official receipts. It excludes competition
files, external data, public-notebook copies, model artifacts, and submissions.
Use [DATA_SOURCES.md](DATA_SOURCES.md) to obtain inputs from their original
sources, and [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md) for the release boundary.

Predict `Will_Buy_EV` with ROC AUC. The first baseline uses only the official competition files, excludes the sequential `id` field, and evaluates CatBoost, LightGBM, XGBoost, their probability mean, and their rank mean under shuffled stratified out-of-fold validation.

## Hard gates

1. Exact official schema, row count, ID order, finite probabilities, and SHA-256 receipts.
2. No target in test, no test hand-labeling, and no hidden/private label source.
3. Validation stability recorded before submission; only one candidate is selected by OOF AUC.
4. External data remains disabled until public accessibility, license, provenance, and rule compliance are documented.

## Reproduce

```bash
python src/audit_data.py \
  --data-dir /private/path/to/official-data \
  --output reports/data_audit.json

python src/train_baseline.py \
  --data-dir /private/path/to/official-data \
  --output-dir /private/path/to/artifacts/baseline_3fold \
  --models lightgbm catboost xgboost \
  --folds 3 --rounds 1400 --seed 20260909

python src/validate_submission.py \
  --sample /private/path/to/official-data/sample_submission.csv \
  --candidate /private/path/to/artifacts/baseline_3fold/submission_rank_blend.csv
```

Official receipts are in `official/`; machine-readable audits and the experiment ledger are in `reports/`.

## First verified submission

Submission `56107927` completed with Public LB `0.94636`. It is the arithmetic mean of two local LightGBM reproductions on identical 5-fold splits; their OOF AUC gap is `0.0000222`, and the mean OOF AUC is `0.9460814`. No third-party OOF predictions were imported.

## Campaign status

Archived on 2026-09-17. Submission `56107927` remains frozen as the final campaign entry; no further training or submission is authorized. See [`reports/campaign_closeout_2026-09-17.md`](reports/campaign_closeout_2026-09-17.md).

## Cite this repository

Please cite the repository snapshot and the immutable commit or tag you used.

```bibtex
@software{openkaggle_playground_series_s6e9_2026,
  author = {Jah-yee},
  title = {Playground Series S6E9 research archive},
  year = {2026},
  url = {https://github.com/OpenKaggle/playground-series-s6e9-research},
  version = {snapshot-2026-09}
}
```
