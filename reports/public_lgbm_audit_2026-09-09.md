# Public LightGBM baseline audit

## Provenance

- Public notebook: https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637
- Notebook owner/title: `najiama/Pure LGBM Model CV 0.94606 LB 0.94637`
- Notebook visibility: public
- Notebook SHA-256: `53a9fc1bc38b2d613ebb0ef77f2480507a9606ff26e697f85f1363279db50a1a`
- External dataset: https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety
- External dataset visibility/access: public, downloadable at no charge through the official Kaggle API
- External dataset license: CC0-1.0
- External data file SHA-256: `c271a380df51b18177d0a039d54525ca7c3d71500701dc7496714a091df16fae`
- External data size: 10,000 rows

## Leakage and rule audit

- The notebook does not read test labels, hand-label the test set, or use a private label source.
- It uses target means from a public, licensed 10,000-row synthetic source dataset. This is external labeled data, not hidden test labels, and is allowed by the competition rule permitting publicly/equally accessible external data.
- It uses unlabeled test features in frequency encodings. This is transductive preprocessing and is recorded explicitly; it does not expose test labels.
- Fold validation applies target encoding by fitting only on each outer training fold and transforming the outer validation/test rows. Training-fold encodings are internally cross-fitted by `sklearn.preprocessing.TargetEncoder`.
- Four hand-authored generator-artifact flags encode public hypotheses: income equals 30,000; income at least 170,537; income between 38,000 and 42,000; and environmental concern equals 1. They are not test labels, but their dependence on label-informed public analysis warrants a clean ablation.
- `id` is excluded. `Number_of_Cars_Owned` is intentionally dropped.
- The reported public notebook scores are OOF about 0.94606 and leaderboard about 0.94637/0.94638. Local replication is required before relying on those claims.

## Decision gates

- Reproduce with 5-fold shuffled stratified OOF and fixed split seed 42.
- Run two model seeds on the same folds; require absolute OOF AUC difference below 0.00005.
- Require OOF AUC at least 0.9458 before submitting.
- Require public leaderboard at least 0.9462 to continue this family.
- Run a clean ablation without the four hard-coded flags. If the artifact version only works by reverse-engineered randomness/noise, or the clean version trails by more than 0.0003, stop this direction.
- Do not import or blend third-party public OOF prediction files.

## Observed local results

- Artifact flags on, model seed 42: OOF `0.9460543587`.
- Artifact flags on, model seed 20260909: OOF `0.9460765773`.
- Two-seed mean: OOF `0.9460813738`; absolute seed gap `0.0000222186`; Public LB `0.94636` (submission `56107927`).
- Artifact flags off, model seed 42: OOF `0.9460593227`.
- Clean-minus-artifact delta on the same split/model seed: `+0.0000049640`.
- Artifact flags off and external original-data means off, model seed 42: OOF `0.9460518514`.
- Removing external original-data target means changes OOF by only `-0.0000074713` versus the clean run.

The clean version does not trail the artifact version, so the `-0.0003` stop condition is not triggered. Neither the four hard-coded flags nor the external source target means provide measurable OOF benefit; both should remain disabled in future work.
