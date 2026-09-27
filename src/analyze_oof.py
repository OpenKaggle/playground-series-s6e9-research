#!/usr/bin/env python3
"""Slice OOF errors and measure model-family diversity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGET = "Will_Buy_EV"
ID_COL = "id"
TARGET_MAP = {"No": 0, "Yes": 1}


def auc_or_none(y: pd.Series, prediction: pd.Series) -> float | None:
    return float(roc_auc_score(y, prediction)) if y.nunique() == 2 else None


def record_slice(
    frame: pd.DataFrame,
    dimension: str,
    value: str,
    strong_col: str,
    base_col: str,
) -> dict[str, object]:
    strong_auc = auc_or_none(frame[TARGET], frame[strong_col])
    base_auc = auc_or_none(frame[TARGET], frame[base_col])
    return {
        "dimension": dimension,
        "value": value,
        "rows": len(frame),
        "positives": int(frame[TARGET].sum()),
        "target_rate": float(frame[TARGET].mean()),
        "strong_auc": strong_auc,
        "base_rank_blend_auc": base_auc,
        "strong_minus_base_auc": (
            float(strong_auc - base_auc) if strong_auc is not None and base_auc is not None else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--strong-oof", type=Path, required=True)
    parser.add_argument("--base-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    strong = pd.read_parquet(args.strong_oof)
    base = pd.read_parquet(args.base_oof)
    if not train[ID_COL].equals(strong[ID_COL]) or not train[ID_COL].equals(base[ID_COL]):
        raise SystemExit("OOF IDs do not match training rows")

    frame = train.copy()
    frame[TARGET] = frame[TARGET].map(TARGET_MAP).astype("int8")
    frame["strong_prediction"] = strong["prediction"].to_numpy()
    for col in ["lightgbm", "catboost", "xgboost", "mean_blend", "rank_blend"]:
        frame[f"base_{col}"] = base[col].to_numpy()

    slices: list[dict[str, object]] = []
    categorical = [
        "Gender",
        "City_Type",
        "Current_Car_Type",
        "Home_Charging_Possible",
        "Subsidy_Available",
        "Range_Anxiety_Level",
    ]
    for col in categorical:
        for value, part in frame.groupby(col, dropna=False, observed=False):
            slices.append(record_slice(part, col, str(value), "strong_prediction", "base_rank_blend"))

    numeric = [
        "Age",
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
        "Environmental_Concern_Level",
    ]
    for col in numeric:
        bins = pd.qcut(frame[col], q=5, duplicates="drop")
        for value, index in frame.groupby(bins, observed=True).groups.items():
            slices.append(record_slice(frame.loc[index], f"{col}_quintile", str(value), "strong_prediction", "base_rank_blend"))

    artifact_masks = {
        "income_eq_30000": frame["Annual_Income_USD"] == 30000.0,
        "income_ge_170537": frame["Annual_Income_USD"] >= 170537.0,
        "income_38000_to_42000": frame["Annual_Income_USD"].between(38000.0, 42000.0),
        "environmental_concern_eq_1": frame["Environmental_Concern_Level"] == 1,
    }
    for name, mask in artifact_masks.items():
        slices.append(record_slice(frame.loc[mask], "public_artifact_hypothesis", name, "strong_prediction", "base_rank_blend"))

    slice_frame = pd.DataFrame(slices).sort_values(["strong_auc", "rows"], na_position="last")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slice_path = args.output_dir / "oof_slice_auc.csv"
    slice_frame.to_csv(slice_path, index=False)

    prediction_cols = [f"base_{col}" for col in ["lightgbm", "catboost", "xgboost", "mean_blend", "rank_blend"]]
    prediction_cols.append("strong_prediction")
    correlations = frame[prediction_cols].corr().round(9)
    correlation_path = args.output_dir / "oof_prediction_correlations.csv"
    correlations.to_csv(correlation_path)

    valid_slices = slice_frame.dropna(subset=["strong_auc"])
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_auc": {
            "strong_seed_blend": float(roc_auc_score(frame[TARGET], frame["strong_prediction"])),
            "base_lightgbm": float(roc_auc_score(frame[TARGET], frame["base_lightgbm"])),
            "base_catboost": float(roc_auc_score(frame[TARGET], frame["base_catboost"])),
            "base_xgboost": float(roc_auc_score(frame[TARGET], frame["base_xgboost"])),
            "base_rank_blend": float(roc_auc_score(frame[TARGET], frame["base_rank_blend"])),
        },
        "lowest_strong_auc_slices": valid_slices.head(10).to_dict(orient="records"),
        "largest_strong_over_base_gains": valid_slices.sort_values(
            "strong_minus_base_auc", ascending=False
        ).head(10).to_dict(orient="records"),
        "base_model_pairwise_correlations": {
            "lightgbm_catboost": float(frame["base_lightgbm"].corr(frame["base_catboost"])),
            "lightgbm_xgboost": float(frame["base_lightgbm"].corr(frame["base_xgboost"])),
            "catboost_xgboost": float(frame["base_catboost"].corr(frame["base_xgboost"])),
        },
        "files": {"slice_auc": str(slice_path), "prediction_correlations": str(correlation_path)},
        "notes": [
            "All metrics use out-of-fold predictions; no leaderboard feedback is used in the slices.",
            "Slices are descriptive and will not be hand-tuned against test records.",
        ],
    }
    report_path = args.output_dir / "oof_error_analysis.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
