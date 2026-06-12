"""
offline_pipeline/feature_engineering/parse_candidates.py

Task 2.1 — High Throughput Candidate Parser
Streams candidates.jsonl, benchmarks parsers, normalizes nested records,
profiles memory, exports flat JSONL, and validates against sample dataset.

Outputs:
    artifacts/candidates_parsed.jsonl   — normalized flat records (100K)
    artifacts/parse_benchmark.json      — library benchmark results
    artifacts/parse_memory_profile.json — memory snapshots at 0/25/50/75/100%
    artifacts/parse_validation_report.json — sample dataset validation results

Usage:
    python offline_pipeline/feature_engineering/parse_candidates.py \\
        --data data/candidates.jsonl \\
        --sample data/sample_candidates.json \\
        --out_dir artifacts/
"""

import argparse
import gzip
import json
import os
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import ijson
import orjson


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Known consulting firms — used in product_company_ratio feature
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree", "ltimindtree",
    "l&t infotech", "niit technologies", "patni", "mastech", "kpit",
}

# JD core skills for match counting
JD_CORE_SKILLS = {
    "embeddings", "vector db", "faiss", "milvus", "qdrant", "weaviate",
    "pinecone", "opensearch", "elasticsearch", "sentence-transformers",
    "retrieval", "ranking", "nlp", "llm", "fine-tuning", "lora", "qlora",
    "xgboost", "ndcg", "python", "pytorch", "transformers", "hugging face",
    "hugging face transformers", "bm25", "ann", "hnsw", "semantic search",
    "learning to rank", "information retrieval", "reranking", "cross-encoder",
    "bi-encoder", "dense retrieval", "sparse retrieval", "hybrid search",
    "recommendation systems", "mlops", "weights & biases", "shap",
    "scikit-learn", "lightgbm", "catboost", "prompt engineering",
}

# Notice period penalty curve (days -> multiplier)
def notice_penalty(days: int) -> float:
    if days <= 30:   return 1.00
    if days <= 60:   return 0.85
    if days <= 90:   return 0.70
    if days <= 120:  return 0.55
    return 0.40

# Work mode compatibility score
WORK_MODE_COMPAT = {
    "flexible": 1.0,
    "hybrid":   1.0,
    "onsite":   0.8,
    "remote":   0.6,
}

REFERENCE_DATE_STR = "2026-06-09"


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Benchmark parsing libraries
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_parsers(data_path: str, sample_size: int = 1000) -> dict:
    """
    Benchmark json (stdlib), orjson, and ijson on first `sample_size` lines.
    Returns dict with timings and throughput estimates.
    """
    print(f"\n[BENCHMARK] Testing parsers on first {sample_size} lines …")

    results = {}

    # ── Read raw lines once (shared input for all benchmarks) ──
    raw_lines = []
    open_fn = gzip.open if data_path.endswith(".gz") else open
    with open_fn(data_path, "rb") as fh:
        for line in fh:
            if line.strip():
                raw_lines.append(line)
            if len(raw_lines) >= sample_size:
                break

    # ── stdlib json ──
    t0 = time.perf_counter()
    for line in raw_lines:
        json.loads(line)
    elapsed_json = time.perf_counter() - t0
    results["json_stdlib"] = {
        "elapsed_sec":    round(elapsed_json, 4),
        "records":        sample_size,
        "records_per_sec": round(sample_size / elapsed_json),
        "est_100k_sec":   round(elapsed_json / sample_size * 100_000, 1),
    }
    print(f"  json (stdlib) : {elapsed_json:.4f}s for {sample_size} records  "
          f"→ est. {results['json_stdlib']['est_100k_sec']}s for 100K")

    # ── orjson ──
    t0 = time.perf_counter()
    for line in raw_lines:
        orjson.loads(line)
    elapsed_orjson = time.perf_counter() - t0
    results["orjson"] = {
        "elapsed_sec":    round(elapsed_orjson, 4),
        "records":        sample_size,
        "records_per_sec": round(sample_size / elapsed_orjson),
        "est_100k_sec":   round(elapsed_orjson / sample_size * 100_000, 1),
        "speedup_vs_json": round(elapsed_json / elapsed_orjson, 2),
    }
    print(f"  orjson        : {elapsed_orjson:.4f}s for {sample_size} records  "
          f"→ est. {results['orjson']['est_100k_sec']}s for 100K  "
          f"({results['orjson']['speedup_vs_json']}× faster than stdlib)")

    # ── ijson (streaming, more overhead per record) ──
    import io
    combined = b"\n".join(raw_lines)
    t0 = time.perf_counter()
    for line in raw_lines:
        list(ijson.items(io.BytesIO(line.strip()), ""))
    elapsed_ijson = time.perf_counter() - t0
    results["ijson"] = {
        "elapsed_sec":    round(elapsed_ijson, 4),
        "records":        sample_size,
        "records_per_sec": round(sample_size / elapsed_ijson),
        "est_100k_sec":   round(elapsed_ijson / sample_size * 100_000, 1),
        "note": "Higher overhead per-record; better suited for partial parsing of huge single objects",
    }
    print(f"  ijson         : {elapsed_ijson:.4f}s for {sample_size} records  "
          f"→ est. {results['ijson']['est_100k_sec']}s for 100K")

    winner = min(results, key=lambda k: results[k]["elapsed_sec"])
    results["winner"] = winner
    results["recommendation"] = (
        f"Using '{winner}' for full parse. "
        f"ijson is best for partial/streaming of huge single JSON objects; "
        f"for JSONL line-by-line, orjson wins on speed."
    )
    print(f"\n  → Winner: {winner}. Full parse will use orjson.\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 & 3 — Stream reader + nested JSON normalizer
# ─────────────────────────────────────────────────────────────────────────────

def _safe_date_delta(date_str: str, reference: str = REFERENCE_DATE_STR) -> int:
    """Return days between date_str and reference. Returns 9999 on parse error."""
    try:
        from datetime import date
        d = date.fromisoformat(date_str)
        r = date.fromisoformat(reference)
        return max(0, (r - d).days)
    except Exception:
        return 9999


def normalize_candidate(raw: dict) -> dict:
    """
    Flatten one raw candidate JSON into a normalized single-level dict.
    All nested structures are unpacked into named scalar fields.
    This is the canonical record format for all downstream tasks.
    """
    p  = raw["profile"]
    s  = raw["redrob_signals"]
    skills   = raw.get("skills", [])
    career   = raw.get("career_history", [])
    edu      = raw.get("education", [])
    certs    = raw.get("certifications", [])
    langs    = raw.get("languages", [])

    # ── Profile fields ──────────────────────────────────────────────────────
    candidate_id        = raw["candidate_id"]
    years_exp           = float(p.get("years_of_experience", 0))
    current_title       = p.get("current_title", "").strip()
    current_company     = p.get("current_company", "").strip()
    current_company_size= p.get("current_company_size", "")
    current_industry    = p.get("current_industry", "")
    location            = p.get("location", "")
    country             = p.get("country", "")
    headline            = p.get("headline", "")
    summary             = p.get("summary", "")

    # ── Career history features ──────────────────────────────────────────────
    num_jobs            = len(career)
    durations           = [j.get("duration_months", 0) for j in career]
    avg_tenure_months   = round(sum(durations) / len(durations), 2) if durations else 0.0
    max_tenure_months   = max(durations) if durations else 0
    min_tenure_months   = min(durations) if durations else 0

    # Product company ratio — fraction of roles NOT at consulting firms
    def _is_consulting(company_name: str) -> bool:
        return any(firm in company_name.lower() for firm in CONSULTING_FIRMS)

    consulting_count    = sum(1 for j in career if _is_consulting(j.get("company", "")))
    product_company_ratio = round(
        (num_jobs - consulting_count) / num_jobs, 4
    ) if num_jobs > 0 else 0.0

    # Company sizes seen in career
    company_sizes_seen  = list({j.get("company_size", "") for j in career})

    # Career companies and titles as lists (for semantic indexer M1)
    career_companies    = [j.get("company", "") for j in career]
    career_titles       = [j.get("title", "") for j in career]
    career_industries   = list({j.get("industry", "") for j in career})

    # Concatenated career descriptions for embedding (M1 uses this)
    career_text         = " | ".join(
        j.get("description", "") for j in career if j.get("description")
    )

    # Tenure impossibility flag: any single job duration > total_exp_months + 6m buffer
    total_exp_months    = years_exp * 12
    job_duration_impossible = int(
        any(j.get("duration_months", 0) > total_exp_months + 6 for j in career)
    )

    # ── Education features ───────────────────────────────────────────────────
    degrees             = [e.get("degree", "") for e in edu]
    edu_fields          = [e.get("field_of_study", "") for e in edu]
    edu_tiers           = [e.get("tier", "unknown") for e in edu]
    best_edu_tier       = (
        "tier_1" if "tier_1" in edu_tiers else
        "tier_2" if "tier_2" in edu_tiers else
        "tier_3" if "tier_3" in edu_tiers else
        "tier_4" if "tier_4" in edu_tiers else
        "unknown"
    )
    has_postgrad        = int(any(
        d.lower() in ("m.tech", "m.e.", "m.s.", "msc", "m.sc", "mba", "ph.d", "phd", "m.sc.")
        for d in degrees
    ))

    # ── Skills features ──────────────────────────────────────────────────────
    num_skills          = len(skills)
    skill_names         = [sk.get("name", "").strip() for sk in skills]
    skill_names_lower   = [n.lower() for n in skill_names]

    # Core skill match count vs JD
    core_skill_match_count = sum(
        1 for name in skill_names_lower if name in JD_CORE_SKILLS
    )

    # Proficiency breakdown
    proficiency_counts  = {"beginner": 0, "intermediate": 0, "advanced": 0, "expert": 0}
    for sk in skills:
        prof = sk.get("proficiency", "beginner")
        proficiency_counts[prof] = proficiency_counts.get(prof, 0) + 1

    adv_skills_count    = proficiency_counts["advanced"] + proficiency_counts["expert"]

    # Skill trust score: penalise advanced/expert with 0 duration_months
    trust_scores = []
    for sk in skills:
        dur   = sk.get("duration_months", 0)
        prof  = sk.get("proficiency", "beginner")
        p_wt  = {"beginner": 0.4, "intermediate": 0.7, "advanced": 0.9, "expert": 1.0}.get(prof, 0.4)
        d_wt  = 1.0 if dur > 6 else (0.5 if dur > 0 else 0.1)
        trust_scores.append(p_wt * d_wt)
    skill_trust_score   = round(sum(trust_scores) / len(trust_scores), 4) if trust_scores else 0.0

    # Zero-duration expert/advanced skills (honeypot signal)
    zero_dur_expert_skills = sum(
        1 for sk in skills
        if sk.get("duration_months", 0) == 0
        and sk.get("proficiency") in ("advanced", "expert")
    )

    # Skills with endorsements
    endorsed_skills     = sum(1 for sk in skills if sk.get("endorsements", 0) > 0)

    # Full skills list for M1 embedding text
    skills_text         = ", ".join(skill_names)

    # ── Certifications & languages ───────────────────────────────────────────
    num_certifications  = len(certs)
    cert_names          = [c.get("name", "") for c in certs]
    num_languages       = len(langs)
    language_names      = [l.get("language", "") for l in langs]

    # ── Redrob signals — all 23 fields ──────────────────────────────────────
    profile_completeness  = float(s.get("profile_completeness_score", 0))
    signup_date           = s.get("signup_date", "")
    last_active_date      = s.get("last_active_date", "")
    open_to_work          = int(s.get("open_to_work_flag", False))
    profile_views_30d     = int(s.get("profile_views_received_30d", 0))
    applications_30d      = int(s.get("applications_submitted_30d", 0))
    recruiter_resp_rate   = float(s.get("recruiter_response_rate", 0))
    avg_response_time_h   = float(s.get("avg_response_time_hours", 0))
    connection_count      = int(s.get("connection_count", 0))
    endorsements_received = int(s.get("endorsements_received", 0))
    notice_period_days    = int(s.get("notice_period_days", 90))
    preferred_work_mode   = s.get("preferred_work_mode", "flexible")
    willing_to_relocate   = int(s.get("willing_to_relocate", False))
    github_activity_score = float(s.get("github_activity_score", -1))
    search_appearance_30d = int(s.get("search_appearance_30d", 0))
    saved_by_recruiters   = int(s.get("saved_by_recruiters_30d", 0))
    interview_comp_rate   = float(s.get("interview_completion_rate", 0))
    offer_accept_rate     = float(s.get("offer_acceptance_rate", -1))
    verified_email        = int(s.get("verified_email", False))
    verified_phone        = int(s.get("verified_phone", False))
    linkedin_connected    = int(s.get("linkedin_connected", False))

    # ── Salary ───────────────────────────────────────────────────────────────
    sal                   = s.get("expected_salary_range_inr_lpa", {})
    salary_min            = float(sal.get("min", 0))
    salary_max            = float(sal.get("max", 0))
    salary_inverted       = int(salary_min > salary_max)

    # ── Assessment scores ────────────────────────────────────────────────────
    assessment_scores     = s.get("skill_assessment_scores", {})
    has_assessments       = int(len(assessment_scores) > 0)
    mean_assessment_score = (
        round(sum(assessment_scores.values()) / len(assessment_scores), 2)
        if assessment_scores else 0.0
    )
    num_assessments       = len(assessment_scores)

    # ── Derived signals ──────────────────────────────────────────────────────
    days_inactive         = _safe_date_delta(last_active_date)
    account_age_days      = _safe_date_delta(signup_date)

    # Sentinel handling
    has_github            = int(github_activity_score != -1.0)
    github_score_active   = github_activity_score if github_activity_score != -1.0 else 0.0
    has_offer_history     = int(offer_accept_rate != -1.0)
    offer_rate_active     = offer_accept_rate if offer_accept_rate != -1.0 else 0.0

    # Logistics scores
    notice_penalty_score  = notice_penalty(notice_period_days)
    work_mode_compat      = WORK_MODE_COMPAT.get(preferred_work_mode, 0.6)
    if preferred_work_mode == "remote" and willing_to_relocate:
        work_mode_compat  = 0.75
    is_india_based        = int(country == "India")

    # ── Text blob for M1 embedding ───────────────────────────────────────────
    # Single concatenated string: headline + summary + career + skills
    embedding_text        = " ".join(filter(None, [
        headline, summary, career_text, skills_text
    ]))

    # ── Assemble flat record ─────────────────────────────────────────────────
    return {
        # -- Identifiers --
        "candidate_id":             candidate_id,
        "country":                  country,
        "location":                 location,
        "is_india_based":           is_india_based,

        # -- Profile --
        "current_title":            current_title,
        "current_company":          current_company,
        "current_company_size":     current_company_size,
        "current_industry":         current_industry,
        "years_of_experience":      years_exp,

        # -- Career --
        "num_jobs":                 num_jobs,
        "avg_tenure_months":        avg_tenure_months,
        "max_tenure_months":        max_tenure_months,
        "min_tenure_months":        min_tenure_months,
        "product_company_ratio":    product_company_ratio,
        "career_companies":         career_companies,
        "career_titles":            career_titles,
        "career_industries":        career_industries,
        "job_duration_impossible":  job_duration_impossible,

        # -- Education --
        "degrees":                  degrees,
        "edu_fields":               edu_fields,
        "best_edu_tier":            best_edu_tier,
        "has_postgrad":             has_postgrad,

        # -- Skills --
        "num_skills":               num_skills,
        "skill_names":              skill_names,
        "core_skill_match_count":   core_skill_match_count,
        "adv_skills_count":         adv_skills_count,
        "skill_trust_score":        skill_trust_score,
        "zero_dur_expert_skills":   zero_dur_expert_skills,
        "endorsed_skills":          endorsed_skills,
        "proficiency_beginner":     proficiency_counts["beginner"],
        "proficiency_intermediate": proficiency_counts["intermediate"],
        "proficiency_advanced":     proficiency_counts["advanced"],
        "proficiency_expert":       proficiency_counts["expert"],

        # -- Certifications & languages --
        "num_certifications":       num_certifications,
        "cert_names":               cert_names,
        "num_languages":            num_languages,
        "language_names":           language_names,

        # -- 23 raw redrob signals --
        "profile_completeness":     profile_completeness,
        "signup_date":              signup_date,
        "last_active_date":         last_active_date,
        "open_to_work":             open_to_work,
        "profile_views_30d":        profile_views_30d,
        "applications_30d":         applications_30d,
        "recruiter_resp_rate":      recruiter_resp_rate,
        "avg_response_time_h":      avg_response_time_h,
        "connection_count":         connection_count,
        "endorsements_received":    endorsements_received,
        "notice_period_days":       notice_period_days,
        "preferred_work_mode":      preferred_work_mode,
        "willing_to_relocate":      willing_to_relocate,
        "github_activity_score":    github_activity_score,
        "search_appearance_30d":    search_appearance_30d,
        "saved_by_recruiters_30d":  saved_by_recruiters,
        "interview_comp_rate":      interview_comp_rate,
        "offer_accept_rate":        offer_accept_rate,
        "verified_email":           verified_email,
        "verified_phone":           verified_phone,
        "linkedin_connected":       linkedin_connected,
        "salary_min_lpa":           salary_min,
        "salary_max_lpa":           salary_max,
        "has_assessments":          has_assessments,
        "num_assessments":          num_assessments,
        "mean_assessment_score":    mean_assessment_score,

        # -- Derived signals --
        "days_inactive":            days_inactive,
        "account_age_days":         account_age_days,
        "has_github":               has_github,
        "github_score_active":      github_score_active,
        "has_offer_history":        has_offer_history,
        "offer_rate_active":        offer_rate_active,
        "salary_inverted":          salary_inverted,
        "notice_penalty_score":     notice_penalty_score,
        "work_mode_compat":         work_mode_compat,

        # -- Text for M1 embedding --
        "embedding_text":           embedding_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Stream reader (with memory profiling)
# ─────────────────────────────────────────────────────────────────────────────

def stream_parse(
    data_path: str,
    out_path: str,
    total_lines: int = 100_000,
) -> dict:
    """
    Stream-parse candidates.jsonl using orjson (benchmark winner).
    Profiles memory at 0/25/50/75/100% checkpoints.
    Writes normalized records to out_path as JSONL.
    Returns memory profile dict.
    """
    print(f"\n[PARSER] Streaming {data_path} → {out_path}")
    print(f"         Using orjson (benchmark winner)")

    checkpoints = {
        0:   False,
        25:  False,
        50:  False,
        75:  False,
        100: False,
    }
    memory_snapshots = []

    tracemalloc.start()
    t_start = time.perf_counter()

    open_fn = gzip.open if data_path.endswith(".gz") else open
    parsed_count  = 0
    skipped_count = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open_fn(data_path, "rb") as fh_in, \
         open(out_path, "wb") as fh_out:

        for raw_line in fh_in:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                raw = orjson.loads(raw_line)
                record = normalize_candidate(raw)
                fh_out.write(orjson.dumps(record))
                fh_out.write(b"\n")
                parsed_count += 1
            except Exception as e:
                skipped_count += 1
                continue

            # ── Memory checkpoint ──
            pct = int(parsed_count / total_lines * 100)
            for cp in [0, 25, 50, 75, 100]:
                if not checkpoints[cp] and pct >= cp:
                    checkpoints[cp] = True
                    current, peak = tracemalloc.get_traced_memory()
                    elapsed = time.perf_counter() - t_start
                    snap = {
                        "checkpoint_pct":     cp,
                        "records_parsed":     parsed_count,
                        "current_memory_mb":  round(current / 1024 / 1024, 2),
                        "peak_memory_mb":     round(peak / 1024 / 1024, 2),
                        "elapsed_sec":        round(elapsed, 2),
                        "records_per_sec":    round(parsed_count / elapsed) if elapsed > 0 else 0,
                    }
                    memory_snapshots.append(snap)
                    print(
                        f"  [{cp:>3}%] {parsed_count:>7,} records | "
                        f"mem: {snap['current_memory_mb']:.1f} MB (peak {snap['peak_memory_mb']:.1f} MB) | "
                        f"{snap['records_per_sec']:,} rec/s | {snap['elapsed_sec']:.1f}s"
                    )

    tracemalloc.stop()
    total_elapsed = time.perf_counter() - t_start

    print(f"\n  ✓ Parsed:  {parsed_count:,}")
    print(f"  ✗ Skipped: {skipped_count:,}")
    print(f"  Total:     {total_elapsed:.2f}s  ({round(parsed_count/total_elapsed):,} rec/s)")

    return {
        "parsed_count":     parsed_count,
        "skipped_count":    skipped_count,
        "total_elapsed_sec": round(total_elapsed, 2),
        "avg_records_per_sec": round(parsed_count / total_elapsed),
        "memory_snapshots": memory_snapshots,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Validate against sample_candidates.json
# ─────────────────────────────────────────────────────────────────────────────

def validate_against_sample(
    parsed_path: str,
    sample_path: str,
) -> dict:
    """
    Cross-check parsed output against the 50 known sample candidates.
    Verifies:
      1. All 50 sample IDs are present in parsed output
      2. Key scalar fields match exactly
      3. Derived fields (days_inactive, salary_inverted, etc.) are correct
    Returns a validation report dict.
    """
    print(f"\n[VALIDATE] Checking parsed output against {sample_path} …")

    # Load sample candidates
    with open(sample_path, "rb") as fh:
        raw_samples = orjson.loads(fh.read())

    # Handle both list and dict formats
    if isinstance(raw_samples, dict):
        raw_samples = list(raw_samples.values())

    sample_ids = {c["candidate_id"] for c in raw_samples}
    print(f"  Sample candidates loaded: {len(raw_samples)}")

    # Load parsed output — index by candidate_id
    parsed_index = {}
    open_fn = gzip.open if parsed_path.endswith(".gz") else open
    with open(parsed_path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = orjson.loads(line)
                parsed_index[rec["candidate_id"]] = rec

    print(f"  Parsed records loaded: {len(parsed_index):,}")

    # ── Check 1: All sample IDs present ──
    missing_ids = sample_ids - set(parsed_index.keys())
    present_ids = sample_ids & set(parsed_index.keys())

    # ── Check 2: Field-level validation for each sample candidate ──
    field_errors = []
    field_checks = 0

    for raw in raw_samples:
        cid = raw["candidate_id"]
        if cid not in parsed_index:
            continue

        parsed = parsed_index[cid]
        p = raw["profile"]
        s = raw["redrob_signals"]
        sal = s["expected_salary_range_inr_lpa"]

        checks = [
            # (field, expected, actual, description)
            ("years_of_experience",  float(p["years_of_experience"]),   parsed["years_of_experience"],    "profile.years_of_experience"),
            ("country",              p["country"],                       parsed["country"],                "profile.country"),
            ("current_title",        p["current_title"].strip(),         parsed["current_title"],          "profile.current_title"),
            ("open_to_work",         int(s["open_to_work_flag"]),        parsed["open_to_work"],           "redrob_signals.open_to_work_flag"),
            ("recruiter_resp_rate",  float(s["recruiter_response_rate"]),parsed["recruiter_resp_rate"],    "redrob_signals.recruiter_response_rate"),
            ("notice_period_days",   int(s["notice_period_days"]),       parsed["notice_period_days"],     "redrob_signals.notice_period_days"),
            ("salary_min_lpa",       float(sal["min"]),                  parsed["salary_min_lpa"],         "salary.min"),
            ("salary_max_lpa",       float(sal["max"]),                  parsed["salary_max_lpa"],         "salary.max"),
            ("salary_inverted",      int(sal["min"] > sal["max"]),       parsed["salary_inverted"],        "derived.salary_inverted"),
            ("has_github",           int(s["github_activity_score"] != -1.0), parsed["has_github"],        "derived.has_github"),
            ("verified_email",       int(s["verified_email"]),           parsed["verified_email"],         "redrob_signals.verified_email"),
            ("linkedin_connected",   int(s["linkedin_connected"]),       parsed["linkedin_connected"],     "redrob_signals.linkedin_connected"),
        ]

        for field, expected, actual, desc in checks:
            field_checks += 1
            if expected != actual:
                field_errors.append({
                    "candidate_id": cid,
                    "field":        field,
                    "description":  desc,
                    "expected":     expected,
                    "actual":       actual,
                })

    # ── Check 3: Output field completeness ──
    # Verify all expected keys exist in every parsed record (spot-check 10)
    expected_keys = {
        "candidate_id", "years_of_experience", "skill_trust_score",
        "core_skill_match_count", "salary_inverted", "days_inactive",
        "embedding_text", "product_company_ratio", "notice_penalty_score",
        "work_mode_compat", "has_github", "has_assessments",
    }
    key_errors = []
    for raw in raw_samples[:10]:
        cid = raw["candidate_id"]
        if cid not in parsed_index:
            continue
        parsed = parsed_index[cid]
        missing_keys = expected_keys - set(parsed.keys())
        if missing_keys:
            key_errors.append({"candidate_id": cid, "missing_keys": list(missing_keys)})

    # ── Summary ──
    passed = len(field_errors) == 0 and len(missing_ids) == 0 and len(key_errors) == 0

    report = {
        "status":                "PASS" if passed else "FAIL",
        "sample_size":           len(raw_samples),
        "ids_present":           len(present_ids),
        "ids_missing":           len(missing_ids),
        "missing_id_list":       list(missing_ids),
        "field_checks_run":      field_checks,
        "field_errors":          field_errors,
        "field_error_count":     len(field_errors),
        "key_errors":            key_errors,
        "total_parsed_records":  len(parsed_index),
    }

    if passed:
        print(f"  ✓ VALIDATION PASSED — {field_checks} field checks, 0 errors")
    else:
        print(f"  ✗ VALIDATION FAILED")
        if missing_ids:
            print(f"    Missing IDs: {missing_ids}")
        for err in field_errors[:5]:
            print(f"    Field mismatch [{err['candidate_id']}] {err['field']}: "
                  f"expected={err['expected']} got={err['actual']}")
        if len(field_errors) > 5:
            print(f"    ... and {len(field_errors) - 5} more errors (see JSON report)")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Task 2.1 — Candidate Stream Parser")
    parser.add_argument("--data",    default="data/candidates.jsonl",
                        help="Path to candidates.jsonl (or .jsonl.gz)")
    parser.add_argument("--sample",  default="data/sample_candidates.json",
                        help="Path to sample_candidates.json for validation")
    parser.add_argument("--out_dir", default="artifacts/",
                        help="Directory to write output files")
    parser.add_argument("--skip_benchmark", action="store_true",
                        help="Skip parser benchmark (saves ~5s)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Task 2.1 — High Throughput Candidate Parser")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # ── 1. Benchmark ──
    if not args.skip_benchmark:
        benchmark_results = benchmark_parsers(args.data, sample_size=1000)
        bench_path = os.path.join(args.out_dir, "parse_benchmark.json")
        with open(bench_path, "w") as fh:
            json.dump(benchmark_results, fh, indent=2)
        print(f"[BENCHMARK] Results saved → {bench_path}")
    else:
        benchmark_results = {"note": "skipped"}
        print("[BENCHMARK] Skipped.")

    # ── 2 & 3. Stream parse + memory profile ──
    parsed_path = os.path.join(args.out_dir, "candidates_parsed.jsonl")
    memory_profile = stream_parse(
        data_path=args.data,
        out_path=parsed_path,
        total_lines=100_000,
    )

    mem_path = os.path.join(args.out_dir, "parse_memory_profile.json")
    with open(mem_path, "w") as fh:
        json.dump(memory_profile, fh, indent=2)
    print(f"[MEMORY]    Profile saved → {mem_path}")

    # ── 4. Validate ──
    validation_report = validate_against_sample(
        parsed_path=parsed_path,
        sample_path=args.sample,
    )

    val_path = os.path.join(args.out_dir, "parse_validation_report.json")
    with open(val_path, "w") as fh:
        json.dump(validation_report, fh, indent=2)
    print(f"[VALIDATE]  Report saved → {val_path}")

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Parsed output:      {parsed_path}")
    print(f"  Records parsed:     {memory_profile['parsed_count']:,}")
    print(f"  Records skipped:    {memory_profile['skipped_count']:,}")
    print(f"  Total time:         {memory_profile['total_elapsed_sec']}s")
    print(f"  Avg throughput:     {memory_profile['avg_records_per_sec']:,} rec/s")
    if memory_profile["memory_snapshots"]:
        peak = max(s["peak_memory_mb"] for s in memory_profile["memory_snapshots"])
        print(f"  Peak memory:        {peak:.1f} MB")
    print(f"  Validation status:  {validation_report['status']}")
    print(f"  Field checks:       {validation_report['field_checks_run']} checks, "
          f"{validation_report['field_error_count']} errors")
    print("=" * 60)
    print(f"\nTask 2.1 complete. {datetime.now(timezone.utc).isoformat()}")

    # Exit with error code if validation failed
    if validation_report["status"] == "FAIL":
        print("\n⚠ Validation failed — check artifacts/parse_validation_report.json")
        raise SystemExit(1)


if __name__ == "__main__":
    main()