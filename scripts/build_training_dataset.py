"""
scripts/build_training_dataset.py

Task 3.4 — Join teacher labels with engineered numerical features.

Produces a tabular training matrix (X + overall_score target) by inner-joining
teacher labels to the Task 2.3 feature matrix on candidate_id. Excludes
labels flagged as API failures by inspect_teacher_labels.py.

Usage:
    python scripts/build_training_dataset.py \\
        --labels artifacts/labeled_candidates.json \\
        --features artifacts/candidate_features.parquet \\
        --integrity artifacts/teacher_labels_integrity.json \\
        --out artifacts/training_dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow imports when run as `python scripts/build_training_dataset.py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from offline_pipeline.feature_engineering.feature_schema import get_feature_columns
from scripts.inspect_teacher_labels import is_failed_label, load_labels


def load_integrity_report(integrity_path: str) -> dict:
    """Load integrity report; raise if check did not pass."""
    path = Path(integrity_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Integrity report not found: {integrity_path}. "
            "Run scripts/inspect_teacher_labels.py first."
        )
    with path.open(encoding="utf-8") as fh:
        report = json.load(fh)
    if not report.get("integrity_passed", False):
        raise RuntimeError(
            "Teacher label integrity check failed. "
            "Fix or regenerate labels before building the training dataset."
        )
    return report


def build_training_dataset(
    labels_path: str,
    features_path: str,
    integrity_path: str,
    out_path: str,
    report_path: str,
) -> pd.DataFrame:
    """Join labels to features and write the training parquet."""
    load_integrity_report(integrity_path)

    labels_df = load_labels(labels_path)
    n_before = len(labels_df)

    # Drop API-failure placeholders; keep legitimate hard-reject zeros
    valid_mask = ~labels_df.apply(is_failed_label, axis=1)
    labels_df = labels_df.loc[valid_mask].reset_index(drop=True)
    excluded_failed = n_before - len(labels_df)

    print(f"\n[LOAD] Reading feature matrix from {features_path} …")
    features_df = pd.read_parquet(features_path)
    print(f"  Loaded {len(features_df):,} candidates, {len(features_df.columns)} columns")

    feature_cols = get_feature_columns()
    missing_cols = [c for c in feature_cols if c not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Feature matrix missing columns: {missing_cols}")

    merged = labels_df.merge(
        features_df[["candidate_id"] + feature_cols],
        on="candidate_id",
        how="inner",
    )

    unmatched = len(labels_df) - len(merged)
    if unmatched > 0:
        raise RuntimeError(
            f"{unmatched} labeled candidates have no matching row in the feature matrix"
        )

    # Final training columns: identifier, numerical features, target
    train_df = merged[["candidate_id"] + feature_cols + ["overall_score"]].copy()

    feature_missing = int(train_df[feature_cols].isna().sum().sum())
    target_missing = int(train_df["overall_score"].isna().sum())
    if feature_missing > 0 or target_missing > 0:
        raise RuntimeError(
            f"Training dataset has missing values "
            f"(features={feature_missing}, target={target_missing})"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out, index=False)

    build_report = {
        "n_labels_input": n_before,
        "n_failed_labels_excluded": excluded_failed,
        "n_training_samples": len(train_df),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "target_column": "overall_score",
        "unmatched_candidate_ids": unmatched,
        "feature_missing_values": feature_missing,
        "target_missing_values": target_missing,
        "target_min": float(train_df["overall_score"].min()),
        "target_max": float(train_df["overall_score"].max()),
        "target_mean": round(float(train_df["overall_score"].mean()), 4),
        "output_path": str(out),
    }

    report_out = Path(report_path)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("w", encoding="utf-8") as fh:
        json.dump(build_report, fh, indent=2)

    print(f"\n=== Training Dataset Built ===")
    print(f"  Input labels:          {n_before:,}")
    print(f"  Failed labels excluded: {excluded_failed:,}")
    print(f"  Training samples:      {len(train_df):,}")
    print(f"  Features:              {len(feature_cols)}")
    print(f"  Target range:          "
          f"{build_report['target_min']:.2f} – {build_report['target_max']:.2f}")
    print(f"  Output:                {out}")
    print("=" * 40)

    return train_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join teacher labels with engineered features for ranker training"
    )
    parser.add_argument(
        "--labels",
        default="artifacts/labeled_candidates.json",
        help="Path to teacher labels JSON",
    )
    parser.add_argument(
        "--features",
        default="artifacts/candidate_features.parquet",
        help="Path to engineered feature matrix",
    )
    parser.add_argument(
        "--integrity",
        default="artifacts/teacher_labels_integrity.json",
        help="Path to integrity report from inspect_teacher_labels.py",
    )
    parser.add_argument(
        "--out",
        default="artifacts/training_dataset.parquet",
        help="Output path for training parquet",
    )
    parser.add_argument(
        "--report",
        default="artifacts/training_dataset_report.json",
        help="Path to write build summary JSON",
    )
    args = parser.parse_args()

    build_training_dataset(
        labels_path=args.labels,
        features_path=args.features,
        integrity_path=args.integrity,
        out_path=args.out,
        report_path=args.report,
    )
    print("\nTask 3.4 preprocessing complete.")


if __name__ == "__main__":
    main()