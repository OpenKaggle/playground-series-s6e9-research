#!/usr/bin/env python3
"""Audit the official Playground S6E9 files without modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "Will_Buy_EV"
ID_COL = "id"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    sample_path = args.data_dir / "sample_submission.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    feature_cols = [col for col in train.columns if col not in {ID_COL, TARGET}]
    train_hash = pd.util.hash_pandas_object(train[feature_cols], index=False)
    test_hash = pd.util.hash_pandas_object(test[feature_cols], index=False)
    duplicate_train_mask = train_hash.duplicated(keep=False)
    duplicate_groups = pd.DataFrame(
        {"feature_hash": train_hash[duplicate_train_mask], "target": train.loc[duplicate_train_mask, TARGET]}
    ).groupby("feature_hash", sort=False)["target"].nunique()
    overlap_hashes = set(train_hash.unique()).intersection(set(test_hash.unique()))

    target_counts = train[TARGET].value_counts(dropna=False)
    target_rates = train[TARGET].value_counts(normalize=True, dropna=False)
    expected_train_features = [col for col in train.columns if col != TARGET]

    checks = {
        "train_schema_matches_test_plus_target": expected_train_features == list(test.columns),
        "sample_schema_exact": list(sample.columns) == [ID_COL, TARGET],
        "sample_row_count_matches_test": len(sample) == len(test),
        "sample_ids_match_test_in_order": sample[ID_COL].equals(test[ID_COL]),
        "train_ids_unique": bool(train[ID_COL].is_unique),
        "test_ids_unique": bool(test[ID_COL].is_unique),
        "train_test_ids_disjoint": not bool(set(train[ID_COL]).intersection(set(test[ID_COL]))),
        "target_binary": set(train[TARGET].dropna().astype(str)) == {"No", "Yes"},
        "target_absent_from_test": TARGET not in test.columns,
        "sample_predictions_finite": bool(np.isfinite(pd.to_numeric(sample[TARGET], errors="coerce")).all()),
        "sample_predictions_in_unit_interval": bool(pd.to_numeric(sample[TARGET], errors="coerce").between(0, 1).all()),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Kaggle competition API download: playground-series-s6e9",
        "license": "CC BY 4.0 (per competition rules)",
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (train_path, test_path, sample_path)
        },
        "rows": {"train": len(train), "test": len(test), "sample_submission": len(sample)},
        "columns": {
            "train": list(train.columns),
            "test": list(test.columns),
            "sample_submission": list(sample.columns),
        },
        "dtypes": {col: str(dtype) for col, dtype in train.dtypes.items()},
        "missing": {
            "train": {col: int(value) for col, value in train.isna().sum().items()},
            "test": {col: int(value) for col, value in test.isna().sum().items()},
        },
        "unique_values_train": {col: int(train[col].nunique(dropna=False)) for col in train.columns},
        "target": {
            "counts": {str(key): int(value) for key, value in target_counts.items()},
            "rates": {str(key): float(value) for key, value in target_rates.items()},
        },
        "id_ranges": {
            "train": [scalar(train[ID_COL].min()), scalar(train[ID_COL].max())],
            "test": [scalar(test[ID_COL].min()), scalar(test[ID_COL].max())],
        },
        "duplicate_feature_audit": {
            "train_rows_in_duplicate_groups": int(duplicate_train_mask.sum()),
            "train_duplicate_groups": int(len(duplicate_groups)),
            "train_duplicate_groups_with_conflicting_targets": int((duplicate_groups > 1).sum()),
            "unique_feature_vectors_shared_between_train_and_test": int(len(overlap_hashes)),
        },
        "categorical_levels": {
            col: sorted(str(value) for value in pd.concat([train[col], test[col]]).dropna().unique())
            for col in feature_cols
            if not pd.api.types.is_numeric_dtype(train[col])
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["all_checks_pass"]:
        raise SystemExit("Data audit failed one or more hard checks")


if __name__ == "__main__":
    main()
