"""
scripts/inspect_teacher_labels.py

Task 3.4 — Teacher label integrity check.

Inspects the LLM-generated teacher labels before model training. Detects
likely API-failure placeholders (all-zero scores with no evidence) and
reports dataset statistics so corrupted labels are not silently used.

Usage:
    python scripts/inspect_teacher_labels.py \\
        --labels artifacts/labeled_candidates.json \\
        --report artifacts/teacher_labels_integrity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCORE_FIELDS = (
    "overall_score",
    "technical_fit",
    "production_fit",
    "career_fit",
    "availability_fit",
    "credibility",
)
SUBSCORE_FIELDS = SCORE_FIELDS[1:]


def load_labels(labels_path: str) -> pd.DataFrame:
    """Load teacher labels JSON array into a DataFrame."""
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Teacher labels not found: {labels_path}")

    with path.open(encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        raise ValueError("Expected labeled_candidates.json to contain a JSON array")

    return pd.DataFrame(records)


def is_failed_label(record: dict) -> bool:
    """
    Heuristic for API timeout / default-template failures.

    Failed generations typically return all-zero scores with empty evidence
    and concerns lists (see generate_labels.py default fallbacks).
    """
    if record.get("overall_score") != 0.0:
        return False
    if any(record.get(field) != 0.0 for field in SUBSCORE_FIELDS):
        return False
    evidence = record.get("evidence") or []
    concerns = record.get("concerns") or []
    return len(evidence) == 0 and len(concerns) == 0


def inspect_labels(df: pd.DataFrame, max_zero_pct: float) -> dict:
    """Compute integrity statistics and pass/fail verdict."""
    n_samples = len(df)
    if n_samples == 0:
        raise ValueError("Teacher labels file is empty")

    duplicate_ids = int(df["candidate_id"].duplicated().sum())
    missing_candidate_id = int(df["candidate_id"].isna().sum())

    score_missing = {
        field: int(df[field].isna().sum()) if field in df.columns else n_samples
        for field in SCORE_FIELDS
    }
    total_score_missing = sum(score_missing.values())

    zero_scores = int((df["overall_score"] == 0.0).sum())
    zero_pct = 100.0 * zero_scores / n_samples

    failed_mask = df.apply(is_failed_label, axis=1)
    failed_count = int(failed_mask.sum())
    failed_pct = 100.0 * failed_count / n_samples

    # Ambiguous zeros: all sub-scores zero but no positive evidence text
    ambiguous_mask = (
        (df["overall_score"] == 0.0)
        & df[list(SUBSCORE_FIELDS)].eq(0.0).all(axis=1)
        & df["evidence"].apply(lambda x: len(x or []) == 0)
        & ~failed_mask
    )
    ambiguous_count = int(ambiguous_mask.sum())

    hard_reject_zeros = int(
        ((df["overall_score"] == 0.0) & df["hard_reject"].fillna(False)).sum()
    )

    passed = (
        duplicate_ids == 0
        and missing_candidate_id == 0
        and total_score_missing == 0
        and failed_pct <= max_zero_pct
    )

    summary = {
        "n_samples": n_samples,
        "zero_score_count": zero_scores,
        "zero_score_pct": round(zero_pct, 2),
        "hard_reject_zero_count": hard_reject_zeros,
        "failed_label_count": failed_count,
        "failed_label_pct": round(failed_pct, 2),
        "ambiguous_zero_count": ambiguous_count,
        "duplicate_candidate_ids": duplicate_ids,
        "missing_candidate_id": missing_candidate_id,
        "missing_score_values": score_missing,
        "total_missing_score_values": total_score_missing,
        "score_min": float(df["overall_score"].min()),
        "score_max": float(df["overall_score"].max()),
        "score_mean": round(float(df["overall_score"].mean()), 4),
        "integrity_passed": passed,
        "max_allowed_failed_pct": max_zero_pct,
    }
    return summary


def print_summary(summary: dict) -> None:
    """Print a human-readable integrity report."""
    print("\n=== Teacher Label Integrity Summary ===")
    print(f"  Samples:                 {summary['n_samples']:,}")
    print(f"  Zero overall_score:      {summary['zero_score_count']:,} "
          f"({summary['zero_score_pct']:.1f}%)")
    print(f"    └ hard_reject zeros:   {summary['hard_reject_zero_count']:,}")
    print(f"  Failed labels (API):   {summary['failed_label_count']:,} "
          f"({summary['failed_label_pct']:.1f}%)")
    print(f"  Ambiguous zero labels:   {summary['ambiguous_zero_count']:,}")
    print(f"  Duplicate candidate_id:  {summary['duplicate_candidate_ids']}")
    print(f"  Missing candidate_id:    {summary['missing_candidate_id']}")
    print(f"  Missing score values:    {summary['total_missing_score_values']}")
    print(f"  Score range:             "
          f"{summary['score_min']:.2f} – {summary['score_max']:.2f} "
          f"(mean {summary['score_mean']:.2f})")
    status = "PASS" if summary["integrity_passed"] else "FAIL"
    print(f"\n  Integrity check:         {status}")
    if not summary["integrity_passed"]:
        print("  Training should not proceed until failed labels are addressed.")
    print("=" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect teacher label integrity before ranker training"
    )
    parser.add_argument(
        "--labels",
        default="artifacts/labeled_candidates.json",
        help="Path to teacher labels JSON",
    )
    parser.add_argument(
        "--report",
        default="artifacts/teacher_labels_integrity.json",
        help="Path to write integrity JSON report",
    )
    parser.add_argument(
        "--max-failed-pct",
        type=float,
        default=5.0,
        help="Maximum allowed percentage of failed (API) labels",
    )
    args = parser.parse_args()

    df = load_labels(args.labels)
    summary = inspect_labels(df, max_zero_pct=args.max_failed_pct)
    print_summary(summary)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nReport saved to {report_path}")

    if not summary["integrity_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
