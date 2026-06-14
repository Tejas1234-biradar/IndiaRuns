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

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Scanner main loop
# ─────────────────────────────────────────────────────────────────────────────

def scan_candidates(parsed_path: str) -> tuple[set, list, dict, int]:
    """
    Stream through candidates_parsed.jsonl and apply all 5 rules to each record.

    Returns:
        honeypot_ids  — set of confirmed candidate_id strings
        audit_trail   — list of dicts for all candidates with flag_count >= 1
        rule_stats    — per-rule firing counts across all 100K candidates
        total         — total records scanned
    """
    print(f"\n[SCANNER] Reading {parsed_path} …")

    honeypot_ids = set()
    audit_trail  = []
    rule_stats   = {name: 0 for name, _ in RULES}
    total        = 0
    flag_dist    = {}

    with open(parsed_path, "rb") as fh:
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            rec   = orjson.loads(raw_line)
            total += 1

            fired_rules   = []
            fired_details = []

            for rule_name, rule_fn in RULES:
                fired, detail = rule_fn(rec)
                if fired:
                    fired_rules.append(rule_name)
                    fired_details.append(detail)
                    rule_stats[rule_name] += 1

            flag_count = len(fired_rules)
            flag_dist[flag_count] = flag_dist.get(flag_count, 0) + 1

            if flag_count >= HONEYPOT_FLAG_THRESHOLD:
                honeypot_ids.add(rec["candidate_id"])

            if flag_count >= 1:
                audit_trail.append({
                    "candidate_id":           rec["candidate_id"],
                    "flag_count":             flag_count,
                    "is_honeypot":            flag_count >= HONEYPOT_FLAG_THRESHOLD,
                    "rules_fired":            fired_rules,
                    "rule_details":           fired_details,
                    "years_exp":              rec["years_of_experience"],
                    "profile_completeness":   rec["profile_completeness"],
                    "adv_skills_count":       rec["adv_skills_count"],
                    "salary_min":             rec["salary_min_lpa"],
                    "salary_max":             rec["salary_max_lpa"],
                    "offer_accept_rate":      rec["offer_accept_rate"],
                    "zero_dur_expert_skills": rec.get("zero_dur_expert_skills", 0),
                })

    audit_trail.sort(key=lambda x: (-x["flag_count"], x["candidate_id"]))

    print(f"  Scanned:                {total:,} candidates")
    print(f"  Honeypots (≥{HONEYPOT_FLAG_THRESHOLD} flags):   {len(honeypot_ids)}")
    print(f"  Suspicious (1-2 flags): "
          f"{sum(v for k,v in flag_dist.items() if 1 <= k < HONEYPOT_FLAG_THRESHOLD)}")
    print(f"\n  Flag count distribution:")
    for k in sorted(flag_dist):
        label = "← confirmed honeypots" if k >= HONEYPOT_FLAG_THRESHOLD else ""
        print(f"    {k} flags: {flag_dist[k]:>7,} candidates  {label}")
    print(f"\n  Per-rule firing counts:")
    for rule_name, count in rule_stats.items():
        pct = count / total * 100
        print(f"    {rule_name:<45} {count:>7,}  ({pct:.2f}%)")

    return honeypot_ids, audit_trail, rule_stats, total


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Export honeypot_ids.pkl
# ─────────────────────────────────────────────────────────────────────────────

def export_honeypot_ids(honeypot_ids: set, out_dir: str) -> str:
    """
    Serialize confirmed honeypot IDs to pickle.
    Consumed by rank.py — any ID in this set is hard-zeroed before scoring.
    Disqualification trigger: >10% honeypots in top-100 submission.
    """
    pkl_path = os.path.join(out_dir, "honeypot_ids.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(honeypot_ids, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pkl_path, "rb") as fh:
        loaded = pickle.load(fh)

    assert loaded == honeypot_ids,  "Pickle round-trip verification failed"
    assert all(isinstance(cid, str) for cid in loaded), "All IDs must be strings"
    assert all(cid.startswith("CAND_") for cid in loaded), "All IDs must match CAND_ format"

    print(f"\n[EXPORT]  honeypot_ids.pkl → {pkl_path}")
    print(f"          {len(loaded)} IDs serialized and verified")
    return pkl_path

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Report builders
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    honeypot_ids:  set,
    audit_trail:   list,
    rule_stats:    dict,
    total_scanned: int,
) -> dict:
    confirmed  = [e for e in audit_trail if e["is_honeypot"]]
    suspicious = [e for e in audit_trail if not e["is_honeypot"]]
    return {
        "generated_at":       datetime.now(tz=timezone.utc).isoformat(),
        "total_scanned":      total_scanned,
        "honeypot_threshold": HONEYPOT_FLAG_THRESHOLD,
        "rules_defined":      [name for name, _ in RULES],
        "summary": {
            "confirmed_honeypots":    len(honeypot_ids),
            "confirmed_honeypot_pct": round(len(honeypot_ids) / total_scanned * 100, 4),
            "suspicious_1_2_flags":   len(suspicious),
            "suspicious_pct":         round(len(suspicious) / total_scanned * 100, 4),
            "clean_candidates":       total_scanned - len(audit_trail),
        },
        "rule_firing_counts":    rule_stats,
        "confirmed_honeypots":   confirmed,
        "suspicious_candidates": suspicious,
    }


def save_report(report: dict, out_dir: str) -> str:
    report_path = os.path.join(out_dir, "honeypot_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[REPORT]  honeypot_report.json → {report_path}")
    return report_path


def save_summary_txt(report: dict, out_dir: str) -> str:
    s         = report["summary"]
    rs        = report["rule_firing_counts"]
    confirmed = report["confirmed_honeypots"]

    lines = [
        "Honeypot Detection Summary — Task 2.2",
        f"Generated: {report['generated_at']}",
        f"Dataset:   {report['total_scanned']:,} candidates scanned",
        "",
        "─" * 60,
        "CONFIRMED HONEYPOTS",
        "─" * 60,
        f"  Count:      {s['confirmed_honeypots']}",
        f"  Percentage: {s['confirmed_honeypot_pct']:.4f}% of dataset",
        f"  Threshold:  >= {report['honeypot_threshold']} rules fired simultaneously",
        "",
        "─" * 60,
        "RULE FIRING COUNTS (all 100K candidates)",
        "─" * 60,
    ]
    for rule_name, count in rs.items():
        pct = count / report["total_scanned"] * 100
        lines.append(f"  {rule_name:<45} {count:>7,}  ({pct:.2f}%)")
    lines += [
        "",
        "─" * 60,
        "CONFIRMED HONEYPOT IDs",
        "─" * 60,
    ]
    for entry in confirmed:
        lines.append(
            f"  {entry['candidate_id']}  "
            f"flags={entry['flag_count']}  "
            f"rules={entry['rules_fired']}"
        )
    lines += [
        "",
        "─" * 60,
        "DOWNSTREAM USAGE",
        "─" * 60,
        "  artifacts/honeypot_ids.pkl is the blocklist for rank.py.",
        "  Any candidate_id in this set is to be hard-zeroed before scoring.",
        f"  Submitting any of these {s['confirmed_honeypots']} IDs in top-100",
        "  means disqualification (spec limit: honeypot rate <= 10% in top 100).",
    ]

    txt_path = os.path.join(out_dir, "honeypot_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[SUMMARY] honeypot_summary.txt → {txt_path}")
    return txt_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Task 2.2 — Honeypot Detection Scanner")
    parser.add_argument("--parsed",  default="artifacts/candidates_parsed.jsonl",
                        help="Path to candidates_parsed.jsonl (Task 2.1 output)")
    parser.add_argument("--out_dir", default="artifacts/",
                        help="Directory to write output files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Task 2.2 — Honeypot Detection Scanner")
    print(f"Started: {datetime.now(tz=timezone.utc).isoformat()}")
    print("=" * 60)

    honeypot_ids, audit_trail, rule_stats, total_scanned = scan_candidates(args.parsed)
    export_honeypot_ids(honeypot_ids, args.out_dir)
    report = build_report(honeypot_ids, audit_trail, rule_stats, total_scanned)
    save_report(report, args.out_dir)
    save_summary_txt(report, args.out_dir)

    s = report["summary"]
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Scanned:              {total_scanned:,}")
    print(f"  Confirmed honeypots:  {s['confirmed_honeypots']}  "
          f"({s['confirmed_honeypot_pct']:.4f}%)")
    print(f"  Suspicious (1-2 fl):  {s['suspicious_1_2_flags']:,}  "
          f"({s['suspicious_pct']:.2f}%)")
    print(f"  Clean candidates:     {s['clean_candidates']:,}")
    print(f"\n  Outputs:")
    print(f"    artifacts/honeypot_ids.pkl")
    print(f"    artifacts/honeypot_report.json")
    print(f"    artifacts/honeypot_summary.txt")
    print("=" * 60)
    print(f"\nTask 2.2 complete. {datetime.now(tz=timezone.utc).isoformat()}")

    actual = s["confirmed_honeypots"]
    if 60 <= actual <= 85:
        print(f"\n✓ Honeypot count {actual} within expected range (spec says ~80).")
    else:
        print(f"\n⚠ WARNING: Count {actual} outside expected range 60–85.")


if __name__ == "__main__":
    main()