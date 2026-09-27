#!/usr/bin/env python3
"""Fail closed when a Playground S6E9 submission violates the official schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "Will_Buy_EV"
ID_COL = "id"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    candidate = pd.read_csv(args.candidate)
    values = pd.to_numeric(candidate[TARGET], errors="coerce") if TARGET in candidate else pd.Series(dtype=float)

    checks = {
        "columns_exact": list(candidate.columns) == [ID_COL, TARGET],
        "row_count_exact": len(candidate) == len(sample),
        "ids_unique": ID_COL in candidate and bool(candidate[ID_COL].is_unique),
        "ids_match_sample_in_order": ID_COL in candidate and candidate[ID_COL].equals(sample[ID_COL]),
        "predictions_numeric": len(values) == len(candidate) and not bool(values.isna().any()),
        "predictions_finite": len(values) == len(candidate) and bool(np.isfinite(values).all()),
        "predictions_in_unit_interval": len(values) == len(candidate) and bool(values.between(0.0, 1.0).all()),
        "predictions_nonconstant": len(values) == len(candidate) and int(values.nunique()) > 1,
    }
    report = {
        "candidate": str(args.candidate),
        "sha256": sha256(args.candidate),
        "rows": len(candidate),
        "prediction_min": float(values.min()) if len(values) else None,
        "prediction_max": float(values.max()) if len(values) else None,
        "prediction_mean": float(values.mean()) if len(values) else None,
        "prediction_std": float(values.std()) if len(values) else None,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    if not report["all_checks_pass"]:
        raise SystemExit("Submission validation failed")


if __name__ == "__main__":
    main()
