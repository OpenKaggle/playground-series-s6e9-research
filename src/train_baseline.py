#!/usr/bin/env python3
"""Reproducible CatBoost/LightGBM/XGBoost OOF baseline for Playground S6E9."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


TARGET = "Will_Buy_EV"
ID_COL = "id"
TARGET_MAP = {"No": 0, "Yes": 1}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def prepare_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    features = [col for col in test.columns if col != ID_COL]
    train_x = train[features].copy()
    test_x = test[features].copy()
    categorical = [col for col in features if not pd.api.types.is_numeric_dtype(train_x[col])]
    for col in categorical:
        categories = sorted(
            set(train_x[col].dropna().astype(str)).union(set(test_x[col].dropna().astype(str)))
        )
        train_x[col] = pd.Categorical(train_x[col].fillna("__MISSING__").astype(str), categories=categories)
        test_x[col] = pd.Categorical(test_x[col].fillna("__MISSING__").astype(str), categories=categories)
    return train_x, test_x, categorical


def build_model(name: str, seed: int, rounds: int):
    if name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=rounds,
            learning_rate=0.035,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.85,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=3.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "catboost":
        return CatBoostClassifier(
            iterations=rounds,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            thread_count=-1,
            allow_writing_files=False,
            verbose=False,
        )
    if name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=rounds,
            learning_rate=0.035,
            max_depth=7,
            min_child_weight=4.0,
            subsample=0.85,
            colsample_bytree=0.90,
            reg_alpha=0.05,
            reg_lambda=3.0,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=120,
        )
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["lightgbm", "catboost", "xgboost"], default=["lightgbm", "catboost", "xgboost"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    sample_path = args.data_dir / "sample_submission.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    y = train[TARGET].map(TARGET_MAP)
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise SystemExit("Unexpected target encoding")

    train_x, test_x, categorical = prepare_frames(train, test)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    fold_records: list[dict[str, object]] = []

    for model_name in args.models:
        model_oof = np.zeros(len(train), dtype=np.float64)
        model_test = np.zeros(len(test), dtype=np.float64)
        for fold, (fit_idx, valid_idx) in enumerate(splitter.split(train_x, y), start=1):
            fold_started = time.time()
            model = build_model(model_name, args.seed + fold, args.rounds)
            fit_x = train_x.iloc[fit_idx]
            valid_x = train_x.iloc[valid_idx]
            fit_y = y.iloc[fit_idx]
            valid_y = y.iloc[valid_idx]

            if model_name == "catboost":
                cat_positions = [train_x.columns.get_loc(col) for col in categorical]
                model.fit(
                    fit_x,
                    fit_y,
                    eval_set=(valid_x, valid_y),
                    cat_features=cat_positions,
                    early_stopping_rounds=120,
                    verbose=False,
                )
            elif model_name == "lightgbm":
                model.fit(
                    fit_x,
                    fit_y,
                    eval_set=[(valid_x, valid_y)],
                    eval_metric="auc",
                    categorical_feature=categorical,
                    callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
                )
            else:
                model.fit(fit_x, fit_y, eval_set=[(valid_x, valid_y)], verbose=False)

            valid_pred = model.predict_proba(valid_x)[:, 1]
            test_pred = model.predict_proba(test_x)[:, 1]
            model_oof[valid_idx] = valid_pred
            model_test += test_pred / args.folds
            best_iteration = getattr(model, "best_iteration_", None)
            fold_auc = roc_auc_score(valid_y, valid_pred)
            fold_record = {
                "model": model_name,
                "fold": fold,
                "auc": float(fold_auc),
                "best_iteration": int(best_iteration) if best_iteration is not None else None,
                "runtime_seconds": round(time.time() - fold_started, 3),
                "fit_rows": len(fit_idx),
                "valid_rows": len(valid_idx),
            }
            fold_records.append(fold_record)
            print(json.dumps(fold_record), flush=True)

        oof[model_name] = model_oof
        test_predictions[model_name] = model_test

    candidate_oof = dict(oof)
    candidate_test = dict(test_predictions)
    if len(args.models) > 1:
        candidate_oof["mean_blend"] = np.mean([oof[name] for name in args.models], axis=0)
        candidate_test["mean_blend"] = np.mean([test_predictions[name] for name in args.models], axis=0)
        candidate_oof["rank_blend"] = np.mean([rank01(oof[name]) for name in args.models], axis=0)
        candidate_test["rank_blend"] = np.mean([rank01(test_predictions[name]) for name in args.models], axis=0)

    candidate_auc = {name: float(roc_auc_score(y, pred)) for name, pred in candidate_oof.items()}
    selected = max(candidate_auc, key=candidate_auc.get)
    selected_test = np.clip(candidate_test[selected], 0.0, 1.0)
    submission = sample.copy()
    submission[TARGET] = selected_test
    submission_path = args.output_dir / f"submission_{selected}.csv"
    submission.to_csv(submission_path, index=False)

    oof_frame = pd.DataFrame({ID_COL: train[ID_COL], TARGET: y})
    for name, pred in candidate_oof.items():
        oof_frame[name] = pred
    oof_path = args.output_dir / "oof_predictions.parquet"
    oof_frame.to_parquet(oof_path, index=False)

    model_fold_auc = {
        name: [float(record["auc"]) for record in fold_records if record["model"] == name]
        for name in args.models
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "official Kaggle competition files only",
        "external_data": False,
        "id_feature_used": False,
        "target": TARGET,
        "metric": "ROC AUC",
        "validation": f"{args.folds}-fold shuffled stratified OOF; seed={args.seed}",
        "models": args.models,
        "rounds_cap": args.rounds,
        "categorical_features": categorical,
        "candidate_oof_auc": candidate_auc,
        "fold_records": fold_records,
        "fold_auc_std": {name: float(np.std(scores)) for name, scores in model_fold_auc.items()},
        "selected_candidate": selected,
        "submission": str(submission_path),
        "submission_sha256": sha256(submission_path),
        "oof_predictions": str(oof_path),
        "runtime_seconds": round(time.time() - started, 3),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "catboost": catboost.__version__,
            "xgboost": xgb.__version__,
        },
        "input_sha256": {path.name: sha256(path) for path in (train_path, test_path, sample_path)},
    }
    report_path = args.output_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
