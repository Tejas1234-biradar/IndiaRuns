"""
tests/test_honeypot.py
Unit tests for all 5 honeypot detection rules.

To run : python -m pytest tests/test_honeypot.py -v
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from offline_pipeline.feature_engineering.detect_honeypots import (
    rule_salary_inversion,
    rule_low_completeness_high_skills,
    rule_salary_inversion_with_offer_history,
    rule_job_duration_impossible,
    rule_zero_duration_expert_cluster,
    HONEYPOT_FLAG_THRESHOLD,
)


# ── record factory ────────────────────────────────────────────────────────────

def make_record(**overrides):
    """Minimal valid parsed record with clean defaults."""
    base = {
        "candidate_id":            "CAND_0000001",
        "salary_min_lpa":          10.0,
        "salary_max_lpa":          20.0,
        "profile_completeness":    70.0,
        "adv_skills_count":        2,
        "offer_accept_rate":       -1.0,
        "years_of_experience":     5.0,
        "job_duration_impossible": 0,
        "zero_dur_expert_skills":  0,
    }
    base.update(overrides)
    return base


# ── Rule 1: salary inversion ──────────────────────────────────────────────────

def test_r1_fires_when_min_greater_than_max():
    rec = make_record(salary_min_lpa=25.0, salary_max_lpa=10.0)
    fired, detail = rule_salary_inversion(rec)
    assert fired is True
    assert "25.0" in detail and "10.0" in detail

def test_r1_does_not_fire_when_valid():
    rec = make_record(salary_min_lpa=10.0, salary_max_lpa=25.0)
    fired, _ = rule_salary_inversion(rec)
    assert fired is False

def test_r1_does_not_fire_when_equal():
    rec = make_record(salary_min_lpa=15.0, salary_max_lpa=15.0)
    fired, _ = rule_salary_inversion(rec)
    assert fired is False


# ── Rule 2: low completeness + high advanced skills ───────────────────────────

def test_r2_fires_on_low_completeness_many_adv_skills():
    rec = make_record(profile_completeness=25.0, adv_skills_count=6)
    fired, detail = rule_low_completeness_high_skills(rec)
    assert fired is True
    assert "25.0" in detail and "6" in detail

def test_r2_does_not_fire_when_completeness_above_threshold():
    rec = make_record(profile_completeness=36.0, adv_skills_count=6)
    fired, _ = rule_low_completeness_high_skills(rec)
    assert fired is False

def test_r2_does_not_fire_when_adv_skills_below_threshold():
    rec = make_record(profile_completeness=25.0, adv_skills_count=4)
    fired, _ = rule_low_completeness_high_skills(rec)
    assert fired is False

def test_r2_fires_at_exact_lower_boundary():
    rec = make_record(profile_completeness=34.9, adv_skills_count=5)
    fired, _ = rule_low_completeness_high_skills(rec)
    assert fired is True

def test_r2_does_not_fire_at_completeness_boundary():
    rec = make_record(profile_completeness=35.0, adv_skills_count=5)
    fired, _ = rule_low_completeness_high_skills(rec)
    assert fired is False


# ── Rule 3: salary inversion with offer history ───────────────────────────────

def test_r3_fires_when_inverted_and_has_offer_history():
    rec = make_record(salary_min_lpa=25.0, salary_max_lpa=10.0, offer_accept_rate=0.75)
    fired, detail = rule_salary_inversion_with_offer_history(rec)
    assert fired is True
    assert "0.75" in detail

def test_r3_does_not_fire_when_no_offer_history():
    rec = make_record(salary_min_lpa=25.0, salary_max_lpa=10.0, offer_accept_rate=-1.0)
    fired, _ = rule_salary_inversion_with_offer_history(rec)
    assert fired is False

def test_r3_does_not_fire_when_salary_valid():
    rec = make_record(salary_min_lpa=10.0, salary_max_lpa=25.0, offer_accept_rate=0.75)
    fired, _ = rule_salary_inversion_with_offer_history(rec)
    assert fired is False

def test_r3_independence_from_r1():
    # R3 requires BOTH salary inversion AND offer history
    # R1 requires only salary inversion — they are distinct checks
    rec_r1_only = make_record(salary_min_lpa=25.0, salary_max_lpa=10.0, offer_accept_rate=-1.0)
    r1_fired, _ = rule_salary_inversion(rec_r1_only)
    r3_fired, _ = rule_salary_inversion_with_offer_history(rec_r1_only)
    assert r1_fired is True
    assert r3_fired is False  # R3 does NOT fire without offer history


# ── Rule 4: job duration impossible ──────────────────────────────────────────

def test_r4_fires_when_flag_set():
    rec = make_record(job_duration_impossible=1, years_of_experience=5.0)
    fired, detail = rule_job_duration_impossible(rec)
    assert fired is True
    assert "5.0" in detail

def test_r4_does_not_fire_when_flag_clear():
    rec = make_record(job_duration_impossible=0)
    fired, _ = rule_job_duration_impossible(rec)
    assert fired is False

def test_r4_detail_contains_exp_months():
    rec = make_record(job_duration_impossible=1, years_of_experience=3.0)
    _, detail = rule_job_duration_impossible(rec)
    assert "36.0" in detail  # 3.0 * 12 = 36.0


# ── Rule 5: zero duration expert cluster ─────────────────────────────────────

def test_r5_fires_at_exactly_3():
    rec = make_record(zero_dur_expert_skills=3)
    fired, detail = rule_zero_duration_expert_cluster(rec)
    assert fired is True
    assert "3" in detail

def test_r5_does_not_fire_at_2():
    rec = make_record(zero_dur_expert_skills=2)
    fired, _ = rule_zero_duration_expert_cluster(rec)
    assert fired is False

def test_r5_fires_above_3():
    rec = make_record(zero_dur_expert_skills=7)
    fired, _ = rule_zero_duration_expert_cluster(rec)
    assert fired is True

def test_r5_does_not_fire_at_zero():
    rec = make_record(zero_dur_expert_skills=0)
    fired, _ = rule_zero_duration_expert_cluster(rec)
    assert fired is False


# ── Threshold constant ────────────────────────────────────────────────────────

def test_honeypot_threshold_is_3():
    assert HONEYPOT_FLAG_THRESHOLD == 3


# ── Multi-rule interaction: confirmed honeypot pattern ────────────────────────

def test_r1_r2_r3_together_would_reach_threshold():
    # The most common honeypot pattern in the dataset
    rec = make_record(
        salary_min_lpa=20.0, salary_max_lpa=10.0,  # R1 + R3
        offer_accept_rate=0.5,                       # R3
        profile_completeness=30.0,                   # R2
        adv_skills_count=7,                          # R2
    )
    r1, _ = rule_salary_inversion(rec)
    r2, _ = rule_low_completeness_high_skills(rec)
    r3, _ = rule_salary_inversion_with_offer_history(rec)
    assert sum([r1, r2, r3]) >= HONEYPOT_FLAG_THRESHOLD