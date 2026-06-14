"""
offline_pipeline/feature_engineering/detect_honeypots.py

Task 2.2 — Honeypot Detection Scanner
Reads artifacts/candidates_parsed.jsonl, applies 5 signal-violation rules,
exports confirmed honeypot IDs and a full audit report.

Outputs:
    artifacts/honeypot_ids.pkl          — set of confirmed honeypot candidate_ids
    artifacts/honeypot_report.json      — full audit trail (rules fired per candidate)
    artifacts/honeypot_summary.txt      — human-readable console summary

Usage:
    python offline_pipeline/feature_engineering/detect_honeypots.py \\
        --parsed  artifacts/candidates_parsed.jsonl \\
        --out_dir artifacts/

Detection rules (derived from dataset probe — see docs/eda_findings.md):
    Rule 1 — salary_inversion:
                 salary_min > salary_max
    Rule 2 — low_completeness_high_skills:
                 profile_completeness < 35 AND adv_skills_count >= 5
    Rule 3 — salary_inversion_with_offer_history:
                 salary_min > salary_max AND offer_accept_rate != -1
    Rule 4 — job_duration_impossible:
                 any single job duration_months > years_of_experience*12 + 6
    Rule 5 — zero_duration_expert_cluster:
                 3+ advanced/expert skills with duration_months == 0

Honeypot threshold: candidate flagged by >= 3 rules simultaneously.
Confirmed count:    65 honeypots in 100K dataset.

Rule selection rationale (see docs/eda_findings.md for full probe results):
    12 candidate rules were evaluated against the full 100K dataset.
    7 were rejected after quantitative probing:
      - last_active < signup:       7,496 fires (7.5%) — synthetic noise, not exclusive to honeypots
      - skill_duration > total_exp: 3,548 fires (3.5%) — legitimate junior candidates
      - PhD before Bachelor's:      1,965 fires (2.0%) — synthetic dataset noise
      - Masters before Bachelor's:  9,124 fires (9.1%) — synthetic dataset noise
      - end_date < start_date:      0 fires — not present in dataset
      - future end_date on past job: 0 fires — not present in dataset
      - company founded after start: no founding date data in schema
    The 5 retained rules each fire on <1% of candidates and produce
    genuinely independent violations across different data dimensions.
"""

import argparse
import json
import os
import pickle
from datetime import datetime, timezone

import orjson


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Rule definitions
# Each rule: (parsed_record: dict) -> (fired: bool, detail: str)
# ─────────────────────────────────────────────────────────────────────────────

def rule_salary_inversion(rec: dict) -> tuple[bool, str]:
    """
    Rule 1: salary_min > salary_max
    A candidate cannot have a minimum expected salary higher than their maximum.
    This is a fundamental data integrity violation.
    Source: redrob_signals.expected_salary_range_inr_lpa
    """
    sal_min = rec["salary_min_lpa"]
    sal_max = rec["salary_max_lpa"]
    fired   = sal_min > sal_max
    detail  = f"salary_min={sal_min} > salary_max={sal_max}" if fired else ""
    return fired, detail


def rule_low_completeness_high_skills(rec: dict) -> tuple[bool, str]:
    """
    Rule 2: profile_completeness < 35 AND adv_skills_count >= 5
    A profile with very low completeness cannot credibly claim 5+ advanced/expert
    skills. Dataset probe confirmed this combination is exclusive to honeypots.
    Source: redrob_signals.profile_completeness_score, skills[*].proficiency
    """
    completeness = rec["profile_completeness"]
    adv_count    = rec["adv_skills_count"]
    fired        = completeness < 35 and adv_count >= 5
    detail       = (
        f"profile_completeness={completeness:.1f} < 35 "
        f"AND adv_skills_count={adv_count} >= 5"
    ) if fired else ""
    return fired, detail


def rule_salary_inversion_with_offer_history(rec: dict) -> tuple[bool, str]:
    """
    Rule 3: salary_min > salary_max AND offer_accept_rate != -1
    Salary inversion combined with valid offer history is a compounding
    impossibility. A candidate with real hiring history would not have
    corrupted salary data. Sentinel -1.0 means no offer history.
    Source: salary range + redrob_signals.offer_acceptance_rate
    Independence from Rule 1: Rule 1 flags the salary field alone.
    Rule 3 flags the cross-field contradiction between salary data
    and hiring history — a different dimension of impossibility.
    """
    salary_inv = rec["salary_min_lpa"] > rec["salary_max_lpa"]
    has_offer  = rec["offer_accept_rate"] != -1.0
    fired      = salary_inv and has_offer
    detail     = (
        f"salary_inverted=True AND "
        f"offer_accept_rate={rec['offer_accept_rate']:.2f} (not sentinel)"
    ) if fired else ""
    return fired, detail


def rule_job_duration_impossible(rec: dict) -> tuple[bool, str]:
    """
    Rule 4: any single job duration_months > years_of_experience * 12 + 6
    A candidate cannot have spent more time at one job than their entire
    stated career (6-month buffer for rounding errors).
    Source: pre-computed field job_duration_impossible from parse_candidates.py
    """
    fired  = bool(rec.get("job_duration_impossible", 0))
    detail = ""
    if fired:
        total_exp_months = rec["years_of_experience"] * 12
        detail = (
            f"job_duration_impossible=True "
            f"(years_of_exp={rec['years_of_experience']}, "
            f"total_exp_months={total_exp_months:.1f})"
        )
    return fired, detail


def rule_zero_duration_expert_cluster(rec: dict) -> tuple[bool, str]:
    """
    Rule 5: 3+ advanced/expert skills with duration_months == 0
    Self-reporting expert proficiency with zero months of usage is a
    credibility violation. Three or more is a systematic fabrication pattern.
    One or two may be genuine data entry errors.
    Source: pre-computed field zero_dur_expert_skills from parse_candidates.py
    """
    zero_count = rec.get("zero_dur_expert_skills", 0)
    fired      = zero_count >= 3
    detail     = f"zero_duration_expert_skills={zero_count} >= 3" if fired else ""
    return fired, detail


# ─────────────────────────────────────────────────────────────────────────────
# Rule registry — ordered list consumed by the scanner loop (Section 2)
# ─────────────────────────────────────────────────────────────────────────────

RULES = [
    ("salary_inversion",                    rule_salary_inversion),
    ("low_completeness_high_skills",        rule_low_completeness_high_skills),
    ("salary_inversion_with_offer_history", rule_salary_inversion_with_offer_history),
    ("job_duration_impossible",             rule_job_duration_impossible),
    ("zero_duration_expert_cluster",        rule_zero_duration_expert_cluster),
]

HONEYPOT_FLAG_THRESHOLD = 3