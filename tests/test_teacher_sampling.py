"""
tests/test_teacher_sampling.py
Unit tests for segment classification and stratified sampling logic.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import pytest

from offline_pipeline.teacher_student.sample_teacher_candidates import (
    classify_segment,
    compute_thresholds,
    stratified_sample,
    validate_sample_diversity,
    exclude_honeypots,
    SAMPLE_ALLOCATION,
)


# ── fixed thresholds for deterministic segment tests ──────────────────────────

T = {
    "faiss_p25": 0.79,
    "faiss_p50": 0.80,
    "faiss_p75": 0.82,
    "skills_p25": 7,
    "skills_p90": 14,
}


def make_row(**overrides):
    base = {
        "faiss_distance_to_jd": 0.80,
        "num_skills_listed":    9,
        "max_assessment_score": 0.0,
    }
    base.update(overrides)
    return pd.Series(base)


# ── Segment classification ────────────────────────────────────────────────────

def test_keyword_stuffer_classification():
    row = make_row(num_skills_listed=18, faiss_distance_to_jd=0.78, max_assessment_score=0.0)
    assert classify_segment(row, T) == "keyword_stuffer"

def test_keyword_stuffer_requires_zero_assessment():
    # Same as above but WITH a validated assessment score -> not a stuffer
    row = make_row(num_skills_listed=18, faiss_distance_to_jd=0.78, max_assessment_score=45.0)
    assert classify_segment(row, T) != "keyword_stuffer"

def test_keyword_stuffer_requires_low_faiss():
    # Many skills but ALSO high faiss distance -> genuinely strong, not stuffing
    row = make_row(num_skills_listed=18, faiss_distance_to_jd=0.90, max_assessment_score=0.0)
    assert classify_segment(row, T) != "keyword_stuffer"

def test_hidden_gem_classification():
    row = make_row(num_skills_listed=5, faiss_distance_to_jd=0.88)
    assert classify_segment(row, T) == "hidden_gem"

def test_hidden_gem_requires_few_skills():
    row = make_row(num_skills_listed=12, faiss_distance_to_jd=0.88)
    assert classify_segment(row, T) != "hidden_gem"

def test_strong_fit_classification():
    row = make_row(num_skills_listed=10, faiss_distance_to_jd=0.85)
    assert classify_segment(row, T) == "strong_fit"

def test_weak_fit_classification():
    row = make_row(num_skills_listed=9, faiss_distance_to_jd=0.70)
    assert classify_segment(row, T) == "weak_fit"

def test_average_match_classification():
    row = make_row(num_skills_listed=9, faiss_distance_to_jd=0.80)
    assert classify_segment(row, T) == "average_match"

def test_stuffer_checked_before_strong_fit():
    # High faiss alone would be strong_fit, but stuffer conditions take priority
    # only when faiss is also <= p50 -- verifies check ordering doesn't
    # misclassify a genuinely strong candidate as a stuffer
    row = make_row(num_skills_listed=18, faiss_distance_to_jd=0.83, max_assessment_score=0.0)
    assert classify_segment(row, T) == "strong_fit"


# ── Honeypot exclusion ────────────────────────────────────────────────────────

def test_exclude_honeypots_removes_exact_set():
    df = pd.DataFrame({
        "candidate_id": ["CAND_0000001", "CAND_0000002", "CAND_0000003"],
        "faiss_distance_to_jd": [0.8, 0.8, 0.8],
    })
    honeypots = {"CAND_0000002"}
    result = exclude_honeypots(df, honeypots)
    assert len(result) == 2
    assert "CAND_0000002" not in result["candidate_id"].values

def test_exclude_honeypots_asserts_on_mismatch():
    df = pd.DataFrame({
        "candidate_id": ["CAND_0000001"],
        "faiss_distance_to_jd": [0.8],
    })
    # Honeypot ID not present in df at all -> removed count (0) != len(honeypots) (1)
    honeypots = {"CAND_9999999"}
    with pytest.raises(AssertionError):
        exclude_honeypots(df, honeypots)


# ── Stratified sampling ───────────────────────────────────────────────────────

def _make_segment_pool(segment, n, start_id=0):
    return pd.DataFrame({
        "candidate_id": [f"CAND_{start_id+i:07d}" for i in range(n)],
        "quality_segment": [segment] * n,
        "faiss_distance_to_jd": np.random.uniform(0.7, 0.9, n),
        "num_skills_listed": np.random.randint(5, 20, n),
    })

def test_stratified_sample_hits_exact_allocation_when_pool_sufficient():
    df = pd.concat([
        _make_segment_pool("strong_fit", 2000, 0),
        _make_segment_pool("average_match", 2000, 2000),
        _make_segment_pool("weak_fit", 2000, 4000),
        _make_segment_pool("hidden_gem", 2000, 6000),
        _make_segment_pool("keyword_stuffer", 2000, 8000),
    ], ignore_index=True)

    sample = stratified_sample(df, SAMPLE_ALLOCATION, seed=1)
    assert len(sample) == sum(SAMPLE_ALLOCATION.values())
    counts = sample["quality_segment"].value_counts().to_dict()
    for seg, target in SAMPLE_ALLOCATION.items():
        assert counts[seg] == target

def test_stratified_sample_no_duplicates():
    df = pd.concat([
        _make_segment_pool("strong_fit", 2000, 0),
        _make_segment_pool("average_match", 2000, 2000),
        _make_segment_pool("weak_fit", 2000, 4000),
        _make_segment_pool("hidden_gem", 2000, 6000),
        _make_segment_pool("keyword_stuffer", 2000, 8000),
    ], ignore_index=True)

    sample = stratified_sample(df, SAMPLE_ALLOCATION, seed=1)
    assert sample["candidate_id"].duplicated().sum() == 0

def test_stratified_sample_handles_shortfall():
    # hidden_gem pool much smaller than its 300 target
    df = pd.concat([
        _make_segment_pool("strong_fit", 2000, 0),
        _make_segment_pool("average_match", 2000, 2000),
        _make_segment_pool("weak_fit", 2000, 4000),
        _make_segment_pool("hidden_gem", 50, 6000),       # shortfall: only 50 available
        _make_segment_pool("keyword_stuffer", 2000, 8000),
    ], ignore_index=True)

    sample = stratified_sample(df, SAMPLE_ALLOCATION, seed=1)
    counts = sample["quality_segment"].value_counts().to_dict()
    # hidden_gem should take all 50 available, not crash
    assert counts["hidden_gem"] == 50
    # total sample size should still be reasonable (redistributed shortfall)
    assert len(sample) <= sum(SAMPLE_ALLOCATION.values())


# ── Diversity validation ──────────────────────────────────────────────────────

def test_validate_diversity_passes_clean_sample():
    df = pd.concat([
        _make_segment_pool("strong_fit", 900, 0),
        _make_segment_pool("average_match", 900, 1000),
        _make_segment_pool("weak_fit", 600, 2000),
        _make_segment_pool("hidden_gem", 300, 3000),
        _make_segment_pool("keyword_stuffer", 300, 4000),
    ], ignore_index=True)

    report = validate_sample_diversity(df, honeypot_ids=set())
    assert report["status"] == "PASS"
    assert report["duplicate_count"] == 0
    assert report["honeypot_leakage"] == []

def test_validate_diversity_fails_on_honeypot_leakage():
    df = _make_segment_pool("strong_fit", 100, 0)
    leaked_id = df["candidate_id"].iloc[0]
    report = validate_sample_diversity(df, honeypot_ids={leaked_id})
    assert report["status"] == "FAIL"
    assert leaked_id in report["honeypot_leakage"]

def test_validate_diversity_fails_on_duplicates():
    df = _make_segment_pool("strong_fit", 50, 0)
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report = validate_sample_diversity(df_with_dupe, honeypot_ids=set())
    assert report["status"] == "FAIL"
    assert report["duplicate_count"] == 1

def test_validate_diversity_fails_on_missing_segment():
    # Only 4 of 5 required segments present
    df = pd.concat([
        _make_segment_pool("strong_fit", 900, 0),
        _make_segment_pool("average_match", 900, 1000),
        _make_segment_pool("weak_fit", 600, 2000),
        _make_segment_pool("hidden_gem", 300, 3000),
        # keyword_stuffer missing entirely
    ], ignore_index=True)
    report = validate_sample_diversity(df, honeypot_ids=set())
    assert report["status"] == "FAIL"
    assert "keyword_stuffer" in str(report["issues"])