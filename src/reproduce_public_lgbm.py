#!/usr/bin/env python3
"""Audited local reproduction of najiama's public S6E9 LightGBM notebook."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


TARGET = "Will_Buy_EV"
ID_COL = "id"
TARGET_MAP = {"No": 0, "Yes": 1}
PUBLIC_NOTEBOOK = "https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engineer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    original: pd.DataFrame | None,
    artifact_flags: bool,
    original_means: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    train = train.copy()
    test = test.copy()
    train[TARGET] = train[TARGET].map(TARGET_MAP)
    if train[TARGET].isna().any():
        raise ValueError("Unexpected target encoding")
    if original_means:
        if original is None:
            raise ValueError("Original data is required when original means are enabled")
        original = original.copy()
        original[TARGET] = original[TARGET].map(TARGET_MAP)
        if original[TARGET].isna().any():
            raise ValueError("Unexpected external target encoding")

    train["is_train"] = 1
    test["is_train"] = 0
    test[TARGET] = np.nan
    combined = pd.concat([train, test], ignore_index=True)
    combined.drop(columns=["Number_of_Cars_Owned"], inplace=True, errors="ignore")

    categorical = combined.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [col for col in combined.columns if col not in categorical + [ID_COL, "is_train", TARGET]]

    digit_features: list[str] = []
    for col in numeric:
        for exponent in range(-4, 4):
            name = f"{col}_digit{exponent}"
            combined[name] = (combined[col].fillna(0) // (10**exponent) % 10).astype("int8")
            digit_features.append(name)
    numeric.extend(digit_features)

    if original_means:
        assert original is not None
        original_global_mean = original[TARGET].mean()
        for col in categorical + numeric:
            if col in original.columns:
                mapping = original.groupby(col, observed=False)[TARGET].mean()
                combined[f"{col}_org_mean"] = combined[col].map(mapping).fillna(original_global_mean).astype("float32")

    numeric_as_categories: list[str] = []
    for col in numeric:
        name = f"{col}_cat"
        combined[name] = combined[col].astype("string").fillna("__MISSING__")
        numeric_as_categories.append(name)

    target_encode_columns = categorical + numeric_as_categories
    for col in target_encode_columns:
        frequency = combined[col].value_counts(normalize=True, dropna=False)
        combined[f"{col}_fe"] = combined[col].map(frequency).fillna(0.0).astype("float32")

    hardcoded_features: list[str] = []
    if artifact_flags:
        hardcoded_features = ["is_30k_spike", "is_millionaire_cliff", "is_dead_zone", "is_env_hater"]
        combined["is_30k_spike"] = (combined["Annual_Income_USD"] == 30000.0).astype("int8")
        combined["is_millionaire_cliff"] = (combined["Annual_Income_USD"] >= 170537.0).astype("int8")
        combined["is_dead_zone"] = (
            (combined["Annual_Income_USD"] >= 38000.0)
            & (combined["Annual_Income_USD"] <= 42000.0)
        ).astype("int8")
        combined["is_env_hater"] = (combined["Environmental_Concern_Level"] == 1).astype("int8")

    combined["income_exact_int"] = np.floor(combined["Annual_Income_USD"]).astype("string")
    combined["income100_floor"] = np.floor(combined["Annual_Income_USD"] / 100.0).astype("string")
    combined["income1000_floor"] = np.floor(combined["Annual_Income_USD"] / 1000.0).astype("string")
    combined["commute_integer"] = np.floor(combined["Daily_Commute_km"]).astype("string")
    target_encode_columns.extend(["income_exact_int", "income100_floor", "income1000_floor", "commute_integer"])

    engineered_train = combined.loc[combined["is_train"] == 1].drop(columns=["is_train"]).copy()
    engineered_test = combined.loc[combined["is_train"] == 0].drop(columns=["is_train", TARGET]).copy()
    del combined
    gc.collect()

    numeric_for_correlation = [
        col
        for col in engineered_train.columns
        if col not in [ID_COL, TARGET] and pd.api.types.is_numeric_dtype(engineered_train[col])
    ]
    corr = engineered_train[numeric_for_correlation].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    perfectly_correlated = [col for col in upper.columns if bool((upper[col] == 1.0).any())]
    constants = [col for col in engineered_train if engineered_train[col].nunique(dropna=False) == 1]
    constants += [col for col in engineered_test if engineered_test[col].nunique(dropna=False) == 1]
    dropped = sorted(set(perfectly_correlated).union(constants) - {ID_COL, TARGET})
    engineered_train.drop(columns=dropped, inplace=True, errors="ignore")
    engineered_test.drop(columns=dropped, inplace=True, errors="ignore")
    target_encode_columns = [col for col in target_encode_columns if col not in dropped]
    features = [col for col in engineered_test.columns if col != ID_COL]
    return engineered_train, engineered_test, features, target_encode_columns, hardcoded_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-flags", choices=["on", "off"], default="on")
    parser.add_argument("--original-means", choices=["on", "off"], default="on")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], default="lightgbm")
    args = parser.parse_args()

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    sample_path = args.data_dir / "sample_submission.csv"
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    original = pd.read_csv(args.original) if args.original_means == "on" else None
    train, test, features, target_encode_columns, hardcoded_features = engineer(
        train_raw,
        test_raw,
        original,
        args.artifact_flags == "on",
        args.original_means == "on",
    )
    X = train[features]
    y = train[TARGET].astype("int8")
    X_test = test[features]
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.split_seed)
    oof = np.zeros(len(train), dtype=np.float64)
    test_prediction = np.zeros(len(test), dtype=np.float64)
    folds: list[dict[str, object]] = []

    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        fold_started = time.time()
        fit_x = X.iloc[fit_idx].copy()
        valid_x = X.iloc[valid_idx].copy()
        test_x = X_test.copy()
        fit_y = y.iloc[fit_idx]
        valid_y = y.iloc[valid_idx]

        for label, smoothing in (("auto", "auto"), ("10", 10.0), ("100", 100.0)):
            encoder = TargetEncoder(
                shuffle=True,
                cv=args.folds,
                smooth=smoothing,
                random_state=args.split_seed,
                target_type="binary",
            )
            encoded_fit = encoder.fit_transform(fit_x[target_encode_columns], fit_y)
            encoded_valid = encoder.transform(valid_x[target_encode_columns])
            encoded_test = encoder.transform(test_x[target_encode_columns])
            for index, col in enumerate(target_encode_columns):
                fit_x[f"{col}_TE_{label}"] = encoded_fit[:, index].astype("float32")
                valid_x[f"{col}_TE_{label}"] = encoded_valid[:, index].astype("float32")
                test_x[f"{col}_TE_{label}"] = encoded_test[:, index].astype("float32")
            del encoder, encoded_fit, encoded_valid, encoded_test
            gc.collect()

        fit_x.drop(columns=target_encode_columns, inplace=True)
        valid_x.drop(columns=target_encode_columns, inplace=True)
        test_x.drop(columns=target_encode_columns, inplace=True)
        if list(fit_x.columns) != list(valid_x.columns) or list(fit_x.columns) != list(test_x.columns):
            raise RuntimeError("Encoded feature schemas diverged")

        if args.model == "lightgbm":
            model = lgb.LGBMClassifier(
                n_estimators=args.rounds,
                learning_rate=0.02,
                max_depth=5,
                num_leaves=32,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.3,
                reg_alpha=0.071,
                reg_lambda=2.0,
                max_bin=1024,
                random_state=args.model_seed,
                feature_pre_filter=False,
                metric="auc",
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(
                fit_x,
                fit_y,
                eval_set=[(valid_x, valid_y)],
                callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)],
            )
        else:
            model = xgb.XGBClassifier(
                n_estimators=args.rounds,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=12.0,
                subsample=0.82,
                colsample_bytree=0.55,
                reg_alpha=0.08,
                reg_lambda=3.0,
                max_bin=1024,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                early_stopping_rounds=150,
                random_state=args.model_seed + fold,
                n_jobs=-1,
            )
            model.fit(fit_x, fit_y, eval_set=[(valid_x, valid_y)], verbose=False)
        valid_prediction = model.predict_proba(valid_x)[:, 1]
        oof[valid_idx] = valid_prediction
        test_prediction += model.predict_proba(test_x)[:, 1] / args.folds
        best_iteration = getattr(model, "best_iteration_", None)
        if best_iteration is None:
            best_iteration = getattr(model, "best_iteration", None)
        record = {
            "fold": fold,
            "auc": float(roc_auc_score(valid_y, valid_prediction)),
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
            "runtime_seconds": round(time.time() - fold_started, 3),
        }
        folds.append(record)
        print(json.dumps(record), flush=True)
        del fit_x, valid_x, test_x, model, valid_prediction
        gc.collect()

    oof_auc = float(roc_auc_score(y, oof))
    variant = "artifact" if args.artifact_flags == "on" else "clean"
    if args.original_means == "off":
        variant += "_noorig"
    submission = sample.copy()
    submission[TARGET] = np.clip(test_prediction, 0.0, 1.0)
    model_suffix = "" if args.model == "lightgbm" else f"_{args.model}"
    submission_path = args.output_dir / f"submission_{variant}{model_suffix}_seed{args.model_seed}.csv"
    submission.to_csv(submission_path, index=False)
    oof_path = args.output_dir / f"oof_{variant}{model_suffix}_seed{args.model_seed}.parquet"
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "prediction": oof}).to_parquet(oof_path, index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_notebook": PUBLIC_NOTEBOOK,
        "variant": variant,
        "model": args.model,
        "artifact_flags": hardcoded_features,
        "competition_data": "official Kaggle files; CC BY 4.0",
        "external_data_used": args.original_means == "on",
        "external_data": (
            {
                "url": "https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety",
                "license": "CC0-1.0",
                "sha256": sha256(args.original),
                "rows": len(original) if original is not None else None,
            }
            if args.original_means == "on"
            else None
        ),
        "test_labels_used": False,
        "transductive_feature_use": "unlabeled train+test frequency encodings only",
        "id_feature_used": False,
        "validation": f"{args.folds}-fold shuffled stratified OOF",
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "folds": folds,
        "oof_auc": oof_auc,
        "fold_auc_std": float(np.std([float(record["auc"]) for record in folds])),
        "features_before_fold_target_encoding": len(features),
        "target_encoded_columns": len(target_encode_columns),
        "runtime_seconds": round(time.time() - started, 3),
        "submission": str(submission_path),
        "submission_sha256": sha256(submission_path),
        "oof": str(oof_path),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
        },
        "input_sha256": {
            train_path.name: sha256(train_path),
            test_path.name: sha256(test_path),
            sample_path.name: sha256(sample_path),
        },
    }
    report_path = args.output_dir / f"report_{variant}{model_suffix}_seed{args.model_seed}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
