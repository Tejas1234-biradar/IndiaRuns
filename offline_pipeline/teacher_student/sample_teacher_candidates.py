"""
offline_pipeline/teacher_student/sample_teacher_candidates.py

Task 2.4 — Diverse Teacher Sampling Pipeline
Builds a stratified sample of 3,000 candidates for LLM Teacher evaluation,
covering strong fits, average matches, weak fits, hidden gems, and
keyword stuffers — while excluding all confirmed honeypots.

Outputs:
    artifacts/teacher_sample.jsonl     — 3,000 sampled candidates with full context
    artifacts/teacher_sample_report.json — segment counts and diversity validation

Usage:
    python offline_pipeline/teacher_student/sample_teacher_candidates.py \\
        --features  artifacts/candidate_features.parquet \\
        --honeypots artifacts/honeypot_ids.pkl \\
        --parsed    artifacts/candidates_parsed.jsonl \\
        --out_dir   artifacts/

Segment definitions (thresholds derived from dataset percentiles):
    Strong fit:       faiss_distance_to_jd >= P75 (0.816), excludes stuffers
    Hidden gem:        faiss_distance_to_jd >= P75 AND num_skills_listed <= P25 (7)
    Keyword stuffer:   num_skills_listed >= P90 (14) AND faiss_distance_to_jd <= P50 (0.804)
                        AND max_assessment_score == 0
    Average match:     faiss_distance_to_jd within [P25, P75], not a stuffer
    Weak fit:          faiss_distance_to_jd <= P25 (0.790)

All segment boundaries computed dynamically from the actual feature matrix
at runtime — not hardcoded — so they remain correct if the matrix changes.
"""

import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_feature_matrix(features_path: str) -> pd.DataFrame:
    """Load the unified feature matrix produced by Task 2.3."""
    print(f"\n[LOAD] Reading feature matrix from {features_path} …")
    df = pd.read_parquet(features_path)
    print(f"  Loaded {len(df):,} candidates, {len(df.columns)} columns")
    return df


def load_honeypot_ids(honeypots_path: str) -> set:
    """Load the confirmed honeypot blocklist from Task 2.2."""
    print(f"[LOAD] Reading honeypot IDs from {honeypots_path} …")
    with open(honeypots_path, "rb") as fh:
        honeypot_ids = pickle.load(fh)
    print(f"  Loaded {len(honeypot_ids)} honeypot IDs")
    return honeypot_ids


def exclude_honeypots(df: pd.DataFrame, honeypot_ids: set) -> pd.DataFrame:
    """Remove all confirmed honeypots from the sampling pool."""
    before = len(df)
    df_clean = df[~df["candidate_id"].isin(honeypot_ids)].reset_index(drop=True)
    removed = before - len(df_clean)
    print(f"[FILTER] Excluded {removed} honeypots from sampling pool "
          f"({before:,} → {len(df_clean):,} candidates)")
    assert removed == len(honeypot_ids), (
        f"Expected to remove {len(honeypot_ids)} honeypots but removed {removed} "
        f"— check for ID mismatches between feature matrix and honeypot pkl"
    )
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Segment classification
# ─────────────────────────────────────────────────────────────────────────────

def compute_thresholds(df: pd.DataFrame) -> dict:
    """
    Compute segment boundary thresholds dynamically from the actual
    distribution of the cleaned (honeypot-free) feature matrix.
    """
    thresholds = {
        "faiss_p25": df["faiss_distance_to_jd"].quantile(0.25),
        "faiss_p50": df["faiss_distance_to_jd"].quantile(0.50),
        "faiss_p75": df["faiss_distance_to_jd"].quantile(0.75),
        "skills_p25": df["num_skills_listed"].quantile(0.25),
        "skills_p90": df["num_skills_listed"].quantile(0.90),
    }
    print("\n[THRESHOLDS] Computed from cleaned feature matrix:")
    for k, v in thresholds.items():
        print(f"    {k}: {v:.4f}")
    return thresholds


def classify_segment(row: pd.Series, t: dict) -> str:
    """
    Classify a single candidate into one of 5 quality segments.
    Order of checks matters — keyword_stuffer and hidden_gem are
    checked before the broader strong/average/weak buckets since
    they represent more specific, higher-priority patterns.
    """
    faiss_dist  = row["faiss_distance_to_jd"]
    num_skills  = row["num_skills_listed"]
    max_assess  = row["max_assessment_score"]

    # Keyword stuffer: many skills listed, low semantic relevance, no validation
    if (num_skills >= t["skills_p90"]
            and faiss_dist <= t["faiss_p50"]
            and max_assess == 0):
        return "keyword_stuffer"

    # Hidden gem: high semantic relevance despite few listed skills
    if faiss_dist >= t["faiss_p75"] and num_skills <= t["skills_p25"]:
        return "hidden_gem"

    # Strong fit: high semantic relevance (and not already a stuffer/gem)
    if faiss_dist >= t["faiss_p75"]:
        return "strong_fit"

    # Weak fit: low semantic relevance
    if faiss_dist <= t["faiss_p25"]:
        return "weak_fit"

    # Everything else: average match
    return "average_match"


def add_segment_column(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Apply segment classification to every candidate in the pool."""
    print("\n[CLASSIFY] Assigning quality segments …")
    df = df.copy()
    df["quality_segment"] = df.apply(
        lambda row: classify_segment(row, thresholds), axis=1
    )
    segment_counts = df["quality_segment"].value_counts()
    print("  Segment distribution (full honeypot-free pool):")
    for seg, count in segment_counts.items():
        pct = count / len(df) * 100
        print(f"    {seg:<20} {count:>7,}  ({pct:.1f}%)")
    return df