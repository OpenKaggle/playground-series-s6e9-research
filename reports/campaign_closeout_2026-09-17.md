# S6E9 campaign closeout — 2026-09-17

## Final state

- Competition: Playground Series S6E9, ROC AUC on `Will_Buy_EV`.
- Frozen final submission: Kaggle submission `56107927`.
- File: `submission_artifact_seedblend.csv`.
- SHA-256: `5d1fd15032b49e934cc6786bd9c1a2590028fd96b6c55d18c42136efb0478e6e`.
- Public leaderboard score: `0.94636`.
- Private leaderboard score: unavailable before competition close.
- Public leader at the final status check: `0.94674` on 2026-09-17.
- Submission count used: 1.
- No further candidate or submission was created on 2026-09-17.
- The campaign heartbeat `ddl-s6e9-4` is paused.

## Why the campaign stopped

The remaining public gap was only `0.00038`, while every legal local extension available from the current pipeline was below the experiment noise floor:

- The frozen two-seed LightGBM parent had OOF AUC `0.9460813738`; its single-seed gap was only `0.0000222186`.
- Removing the hardcoded threshold flags changed OOF by only `+0.0000049640` at the same seed.
- Removing the external source-label means then changed OOF by only `-0.0000074713`.
- A clean official-data-only XGBoost reached `0.9459964563`; its best coarse LightGBM/XGBoost rank blend improved the clean LightGBM by only `0.0000212182`, with prediction correlation `0.999373`.
- Current public research reported that two local blends gaining about `0.00025–0.00026` OOF paid only `-0.00001/+0.00001` on the public board. This is strong evidence against spending more compute on highly correlated microblends.
- Recent advertised `0.94650` tie-breaking work used public prediction inputs plus recovered generator thresholds, so it was ineligible under this campaign's provenance and anti-reconstruction rules.

The 2026-09-17 clean LightGBM second-seed run was interrupted during its first fold immediately after the control plane changed the campaign to archive-only. It produced no OOF vector, test prediction, or submission file.

## Transferable lessons

1. Freeze folds, seeds, data hashes, and submission hashes before comparing small AUC changes. At this scale, a positive fifth decimal is not evidence by itself.
2. Digit channels, high-resolution binning, frequency encoding, and strictly fold-local target encoding were the useful feature family. Plain model-family diversity was not enough.
3. Hardcoded generator boundaries and original-table label priors were not responsible for the strong score; removing them changed OOF by less than `1e-5` in the controlled ablations.
4. A second model is not a useful hedge merely because it is different in name. It must be both close in expected score and materially decorrelated; correlations around `0.999` made the available alternatives effectively inert.
5. Global monotonic calibration cannot improve ROC AUC because it preserves ordering. Group-specific calibration can change ordering, but requires genuinely nested validation; fitting it on the same OOF labels is not acceptable evidence.
6. Public OOF files, public test predictions, scored-submission histories, and leaderboard-derived thresholds are useful for auditing claims but not for constructing a compliant independent candidate.
7. Stop when all remaining routes are micro-optimizations below the measured noise bar. Preserving one well-audited entry is higher value than consuming quota on unrepeatable gains.

## Reopen condition

Do not resume this campaign automatically. Reopen only by explicit user decision and only for a predeclared independent method family that improves the same frozen folds by at least `0.00010` and reproduces across two seeds without public prediction inputs, source-label reconstruction, or leaderboard feedback.
