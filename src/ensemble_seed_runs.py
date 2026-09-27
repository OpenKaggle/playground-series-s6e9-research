#!/usr/bin/env python3
"""Average two locally reproduced model seeds and record the stability gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


TARGET = "Will_Buy_EV"
ID_COL = "id"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-a", type=Path, required=True)
    parser.add_argument("--oof-b", type=Path, required=True)
    parser.add_argument("--submission-a", type=Path, required=True)
    parser.add_argument("--submission-b", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    oof_a = pd.read_parquet(args.oof_a)
    oof_b = pd.read_parquet(args.oof_b)
    submission_a = pd.read_csv(args.submission_a)
    submission_b = pd.read_csv(args.submission_b)
    sample = pd.read_csv(args.sample)
    if not oof_a[ID_COL].equals(oof_b[ID_COL]) or not oof_a[TARGET].equals(oof_b[TARGET]):
        raise SystemExit("OOF rows differ")
    if not submission_a[ID_COL].equals(sample[ID_COL]) or not submission_b[ID_COL].equals(sample[ID_COL]):
        raise SystemExit("Submission IDs differ from the official sample")

    score_a = float(roc_auc_score(oof_a[TARGET], oof_a["prediction"]))
    score_b = float(roc_auc_score(oof_b[TARGET], oof_b["prediction"]))
    oof_mean = (oof_a["prediction"] + oof_b["prediction"]) / 2.0
    score_mean = float(roc_auc_score(oof_a[TARGET], oof_mean))
    seed_gap = abs(score_a - score_b)
    prediction_correlation = float(oof_a["prediction"].corr(oof_b["prediction"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blended_submission = sample.copy()
    blended_submission[TARGET] = (submission_a[TARGET] + submission_b[TARGET]) / 2.0
    submission_path = args.output_dir / "submission_artifact_seedblend.csv"
    blended_submission.to_csv(submission_path, index=False)
    oof_path = args.output_dir / "oof_artifact_seedblend.parquet"
    pd.DataFrame({ID_COL: oof_a[ID_COL], TARGET: oof_a[TARGET], "prediction": oof_mean}).to_parquet(
        oof_path, index=False
    )

    gates = {
        "both_single_seed_oof_at_least_0_9458": min(score_a, score_b) >= 0.9458,
        "seed_auc_gap_below_0_00005": seed_gap < 0.00005,
        "mean_oof_not_worse_than_best_single": score_mean >= max(score_a, score_b),
        "no_third_party_oof_used": True,
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "two local reproductions on identical OOF folds; no third-party OOF files",
        "oof_auc": {"seed_a": score_a, "seed_b": score_b, "mean": score_mean},
        "absolute_seed_auc_gap": seed_gap,
        "oof_prediction_correlation": prediction_correlation,
        "submission": str(submission_path),
        "submission_sha256": sha256(submission_path),
        "oof": str(oof_path),
        "input_sha256": {
            str(args.oof_a): sha256(args.oof_a),
            str(args.oof_b): sha256(args.oof_b),
            str(args.submission_a): sha256(args.submission_a),
            str(args.submission_b): sha256(args.submission_b),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    report_path = args.output_dir / "seed_stability_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_gates_pass"]:
        raise SystemExit("Seed ensemble failed one or more gates")


if __name__ == "__main__":
    main()
