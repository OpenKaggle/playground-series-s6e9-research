# Public notebook scout — 2026-09-09

Scope: official Kaggle CLI downloads only. Public notebooks were audited for methods and provenance. No public OOF or test prediction file was used in local training or ensembling.

## Decisions

- `jazivxt/single-model-zoom-zoom`: reject its final prediction path. The advertised `submission.csv` is 95% an attached scored reference submission and 5% a newly trained model, with additional experiments derived from scored submission history. This is not an independent reproducible model candidate. Its method-only ideas (digit channels, train+test frequency encoding, high-resolution income response estimation) overlap substantially with the existing advanced clean pipeline.
- `najiama/s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633`: reject. The notebook does not train models; it loads a public private-strategy OOF blend and matching test predictions. This violates the campaign rule against public OOF/prediction blending.
- `cdeotte/fable-5-1-xgb-starter`: do not pursue the recipe/base-margin variants. They explicitly encode a recovered synthetic-generator formula. The user's stop condition excludes spending the campaign on generator/RNG reverse engineering. The four ordinary row-wise interaction features are method ideas only and are not expected to add material diversity to the current high-capacity pipeline.
- `georgymamarin/s6e9-starter-how-to-tell-a-real-gain-from-noise`: retain as methodology guidance. Its main promising observations—higher bin resolution, decimal digit channels, and frequency encoding—are already present in the clean LightGBM reproduction. Its noise-control recommendation supports rejecting marginal single-seed gains.
- `mikhailnaumov/electric-vehicle-purchases-single-xgb`: no direct reproduction. The current revision has cleared outputs, a very expensive 10-fold/50,000-round XGBoost path, and many digit/target-encoding features that overlap with the current clean pipeline.
- `kospintr/evehicle-stacked-lgbm-catb-xgb-hgbc-baseline`: no direct reproduction. Its current configuration is a single CatBoost model, includes the hardcoded 38k–42k income dead-zone artifact, and has no preserved score output in the pulled notebook.

## Independent follow-up

The promising non-public-prediction path was tested independently: XGBoost on the official-data-only clean advanced features, using the same five shuffled stratified folds as the LightGBM control.

- XGBoost OOF AUC: `0.9459964562506145`
- LightGBM clean/no-external OOF AUC: `0.9460518513674472`
- OOF correlation: `0.999372957754415`
- Equal rank blend: `0.9460688500591461`
- Best coarse-grid rank blend (65% LightGBM): `0.9460730695549664`
- Gain over LightGBM: `0.0000212181875192`

Decision: reject the blend offline. The gain is well below a defensible noise threshold and lacks a second-seed confirmation. No Kaggle submission was made.

## Source hashes

- `najiama/s6e9-electric-vehicle-oof-cv-0-94618-lb-0-94633`: `e1b8e113906636f67f7e1ed6878b6442a362b5edb7b224899214f3ec081580af`
- `cdeotte/fable-5-1-xgb-starter`: `08cc650a222b62385a18226871a4481c3050f790d01f7006c6562240c815ae96`
- `georgymamarin/s6e9-starter-how-to-tell-a-real-gain-from-noise`: `d6328dfda63b79195241bd5ca3889660aa1142664b7f30ff80646cecddf44985`
- `jazivxt/single-model-zoom-zoom`: `da3571310b5e66d271d80b9ba69aeccae198a6956bd3ade01299f51ca10ab675`
- `mikhailnaumov/electric-vehicle-purchases-single-xgb`: `d3b6a4b89f041ecb3bce657a67f60e5f1526d121bfb46358ea821bc7dc0a3b10`
- `kospintr/evehicle-stacked-lgbm-catb-xgb-hgbc-baseline`: `8b8d7ac00e529322035b5b1ccc9ecdcdb92082b0ef2c824fa9bb7950fa027177`
