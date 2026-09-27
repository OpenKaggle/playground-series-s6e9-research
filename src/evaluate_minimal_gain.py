#!/usr/bin/env python3
"""Evaluate a two-seed clean candidate against a frozen parent without creating a submission."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "Will_Buy_EV"
ID_COL = "id"
TARGET_MAP = {"No": 0, "Yes": 1}


def ece(y: np.ndarray, prediction: np.ndarray, bins: int = 20) -> float:
    """Equal-frequency expected calibration error."""
    order = np.argsort(prediction, kind="mergesort")
    total = len(y)
    value = 0.0
    for index in np.array_split(order, bins):
        if len(index):
            value += len(index) / total * abs(float(y[index].mean() - prediction[index].mean()))
    return float(value)


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    clipped = np.clip(prediction, 1e-15, 1 - 1e-15)
    return {
        "auc": float(roc_auc_score(y, prediction)),
        "log_loss": float(log_loss(y, clipped)),
        "brier": float(brier_score_loss(y, prediction)),
        "ece_20_equal_frequency": ece(y, prediction),
        "mean_prediction": float(prediction.mean()),
        "prevalence": float(y.mean()),
    }


def auc_or_none(y: np.ndarray, prediction: np.ndarray) -> float | None:
    return float(roc_auc_score(y, prediction)) if np.unique(y).size == 2 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--parent-oof", type=Path, required=True)
    parser.add_argument("--candidate-oof-a", type=Path, required=True)
    parser.add_argument("--candidate-oof-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--minimum-gain", type=float, default=0.00005)
    parser.add_argument("--maximum-seed-gap", type=float, default=0.00005)
    parser.add_argument("--resample-shuffles", type=int, default=12)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test, usecols=[ID_COL])
    parent = pd.read_parquet(args.parent_oof)
    candidate_a = pd.read_parquet(args.candidate_oof_a)
    candidate_b = pd.read_parquet(args.candidate_oof_b)
    y = train[TARGET].map(TARGET_MAP).to_numpy(np.int8)

    for name, frame in {
        "parent": parent,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
    }.items():
        if not train[ID_COL].equals(frame[ID_COL]):
            raise SystemExit(f"{name} OOF IDs do not match official training rows")
        if not np.array_equal(y, frame[TARGET].to_numpy(np.int8)):
            raise SystemExit(f"{name} OOF labels do not match official training labels")
        prediction = frame["prediction"].to_numpy(float)
        if not np.isfinite(prediction).all() or not ((prediction >= 0) & (prediction <= 1)).all():
            raise SystemExit(f"{name} predictions are invalid")

    parent_prediction = parent["prediction"].to_numpy(float)
    prediction_a = candidate_a["prediction"].to_numpy(float)
    prediction_b = candidate_b["prediction"].to_numpy(float)
    candidate_mean = (prediction_a + prediction_b) / 2.0

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.split_seed)
    fold_id = np.full(len(train), -1, dtype=np.int8)
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(train)), y)):
        fold_id[held] = fold
    if (fold_id < 0).any():
        raise RuntimeError("Fold reconstruction failed")

    model_predictions = {
        "parent": parent_prediction,
        "candidate_seed_a": prediction_a,
        "candidate_seed_b": prediction_b,
        "candidate_seed_mean": candidate_mean,
    }
    overall = {name: metrics(y, prediction) for name, prediction in model_predictions.items()}
    parent_auc = overall["parent"]["auc"]
    mean_auc = overall["candidate_seed_mean"]["auc"]
    seed_gap = abs(overall["candidate_seed_a"]["auc"] - overall["candidate_seed_b"]["auc"])

    fold_rows: list[dict[str, object]] = []
    for fold in range(5):
        mask = fold_id == fold
        row: dict[str, object] = {"fold": fold, "rows": int(mask.sum()), "positives": int(y[mask].sum())}
        for name, prediction in model_predictions.items():
            row[f"{name}_auc"] = float(roc_auc_score(y[mask], prediction[mask]))
        row["candidate_mean_minus_parent"] = row["candidate_seed_mean_auc"] - row["parent_auc"]
        fold_rows.append(row)

    segment_specs: list[tuple[str, pd.Series]] = []
    categorical = [
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
        "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",
    ]
    for column in categorical:
        segment_specs.append((column, train[column].astype(str)))
    for column in ["Age", "Annual_Income_USD", "Daily_Commute_km"]:
        segment_specs.append((f"{column}_quintile", pd.qcut(train[column], 5, duplicates="drop").astype(str)))

    segment_rows: list[dict[str, object]] = []
    for dimension, values in segment_specs:
        for value in sorted(values.unique()):
            mask = values.to_numpy() == value
            positives = int(y[mask].sum())
            negatives = int(mask.sum() - positives)
            if positives < 100 or negatives < 100:
                continue
            row: dict[str, object] = {
                "dimension": dimension,
                "value": value,
                "rows": int(mask.sum()),
                "positives": positives,
                "negatives": negatives,
                "prevalence": float(y[mask].mean()),
            }
            for name, prediction in model_predictions.items():
                row[f"{name}_auc"] = auc_or_none(y[mask], prediction[mask])
                row[f"{name}_calibration_bias"] = float(prediction[mask].mean() - y[mask].mean())
            row["seed_auc_gap"] = abs(row["candidate_seed_a_auc"] - row["candidate_seed_b_auc"])
            row["candidate_mean_minus_parent"] = row["candidate_seed_mean_auc"] - row["parent_auc"]
            segment_rows.append(row)

    segment_frame = pd.DataFrame(segment_rows).sort_values(["dimension", "value"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.output_dir / "minimal_gain_segment_stability.csv"
    fold_path = args.output_dir / "minimal_gain_fold_stability.csv"
    segment_frame.to_csv(segment_path, index=False)
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    fold_deltas = np.array([float(row["candidate_mean_minus_parent"]) for row in fold_rows])
    slice_deltas = segment_frame["candidate_mean_minus_parent"].to_numpy(float)

    public_rows = int(len(test) * 0.20)
    chunks_per_shuffle = len(train) // public_rows
    rng = np.random.default_rng(args.split_seed)
    resampled_parent: list[float] = []
    resampled_candidate: list[float] = []
    for _ in range(args.resample_shuffles):
        permutation = rng.permutation(len(train))
        for chunk in range(chunks_per_shuffle):
            index = permutation[chunk * public_rows : (chunk + 1) * public_rows]
            resampled_parent.append(float(roc_auc_score(y[index], parent_prediction[index])))
            resampled_candidate.append(float(roc_auc_score(y[index], candidate_mean[index])))
    resampled_parent_array = np.asarray(resampled_parent)
    resampled_candidate_array = np.asarray(resampled_candidate)
    resampled_delta = resampled_candidate_array - resampled_parent_array
    score_stability = {
        "public_rows_assumed": public_rows,
        "shuffles": args.resample_shuffles,
        "disjoint_chunks_per_shuffle": chunks_per_shuffle,
        "scored_chunks": len(resampled_delta),
        "parent_candidate_score_correlation": float(
            np.corrcoef(resampled_parent_array, resampled_candidate_array)[0, 1]
        ),
        "candidate_minus_parent_mean": float(resampled_delta.mean()),
        "candidate_minus_parent_standard_deviation": float(resampled_delta.std(ddof=1)),
        "candidate_better_fraction": float((resampled_delta > 0).mean()),
        "candidate_minus_parent_q05": float(np.quantile(resampled_delta, 0.05)),
        "candidate_minus_parent_q95": float(np.quantile(resampled_delta, 0.95)),
    }
    gates = {
        "candidate_seed_gap_below_limit": bool(seed_gap < args.maximum_seed_gap),
        "candidate_mean_gain_at_least_minimum": bool(mean_auc - parent_auc >= args.minimum_gain),
        "candidate_mean_wins_at_least_four_folds": bool((fold_deltas > 0).sum() >= 4),
        "no_segment_regression_below_minus_0_0003": bool(slice_deltas.min(initial=0.0) >= -0.0003),
        "no_public_prediction_or_external_source_retraining": True,
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_scope": "Diagnostic only; this script never writes a submission.",
        "overall": overall,
        "candidate_seed_auc_gap": seed_gap,
        "candidate_seed_prediction_correlation": float(np.corrcoef(prediction_a, prediction_b)[0, 1]),
        "candidate_mean_prediction_correlation_with_parent": float(
            np.corrcoef(candidate_mean, parent_prediction)[0, 1]
        ),
        "candidate_mean_minus_parent_auc": mean_auc - parent_auc,
        "folds": fold_rows,
        "fold_wins": int((fold_deltas > 0).sum()),
        "segments": {
            "count": int(len(segment_frame)),
            "median_seed_auc_gap": float(segment_frame["seed_auc_gap"].median()),
            "maximum_seed_auc_gap": float(segment_frame["seed_auc_gap"].max()),
            "median_candidate_mean_minus_parent": float(segment_frame["candidate_mean_minus_parent"].median()),
            "minimum_candidate_mean_minus_parent": float(segment_frame["candidate_mean_minus_parent"].min()),
            "maximum_candidate_mean_minus_parent": float(segment_frame["candidate_mean_minus_parent"].max()),
        },
        "resampled_public_slice_stability": score_stability,
        "calibration_policy": {
            "global_monotonic_calibration": "Not promoted: a single monotonic transform preserves ROC-AUC ordering and cannot create a genuine AUC gain.",
            "group_or_fold_specific_calibration": "Not promoted without nested base-model OOF because it can change cross-group ordering and overfit the same labels.",
            "diagnostics": "Log loss, Brier score, ECE, and per-segment mean bias are reported only as stability diagnostics.",
        },
        "thresholds": {
            "minimum_gain": args.minimum_gain,
            "maximum_seed_gap": args.maximum_seed_gap,
        },
        "gates": gates,
        "candidate_authorized": all(gates.values()),
        "files": {"fold_stability": str(fold_path), "segment_stability": str(segment_path)},
    }
    report_path = args.output_dir / "minimal_gain_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
