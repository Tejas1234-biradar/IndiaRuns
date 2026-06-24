import numpy as np
import pandas as pd
from datetime import datetime

# Reference date for recency calculations
REF_DATE = datetime(2026, 6, 22)

REQUIRED_SKILLS = {"embedding", "retrieval", "search", "vector", "python", "ranking", "nlp"}
PREFERRED_SKILLS = {"fine-tuning", "finetuning", "lora", "qlora", "peft", "xgboost", "lightgbm", "ranker", "learning to rank"}

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def extract_candidate_features(candidate):
    """
    Extracts raw features from a single candidate dictionary.
    Returns a dict mapping feature names to raw values (with NaNs for missing/sentinels).
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    cid = candidate.get("candidate_id")
    
    # 1. JD Relevance Proxies
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()
    desc_text = " ".join([j.get("description", "").lower() for j in history])
    skills_lowered = [s.get("name", "").lower() for s in skills]
    
    # Cosine similarity proxy: check relevance keywords in summary, headline, title
    relevance_keywords = ["ai", "machine learning", "ml", "nlp", "search", "retrieval", "embedding", "vector", "deep learning"]
    keyword_matches = sum(1 for kw in relevance_keywords if kw in summary or kw in headline or kw in desc_text)
    embedding_similarity = 0.3 + min(0.6, keyword_matches * 0.1) # scale between 0.3 and 0.9
    
    # Retrieval similarity proxy: check search-specific words
    retrieval_keywords = ["vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch", "hybrid search", "bm25"]
    retrieval_matches = sum(1 for kw in retrieval_keywords if kw in summary or kw in desc_text or any(kw in s for s in skills_lowered))
    retrieval_similarity = min(1.0, retrieval_matches * 0.25)
    
    # Skill matches
    req_matches = sum(1 for s in skills_lowered if any(req in s for req in REQUIRED_SKILLS))
    jd_required_skill_matches = req_matches / len(REQUIRED_SKILLS) if REQUIRED_SKILLS else 0.0
    
    pref_matches = sum(1 for s in skills_lowered if any(pref in s for pref in PREFERRED_SKILLS))
    jd_preferred_skill_matches = pref_matches / len(PREFERRED_SKILLS) if PREFERRED_SKILLS else 0.0
    
    # 2. Career Quality
    years_exp = profile.get("years_of_experience", 0.0)
    
    # Job durations
    job_durations = [j.get("duration_months", 0) for j in history]
    avg_job_tenure_months = np.mean(job_durations) if job_durations else np.nan
    num_jobs = len(history)
    
    short_tenure_count = sum(1 for j in history if j.get("duration_months", 0) < 12 and not j.get("is_current", False))
    is_tier_1_education = float(any(edu.get("tier") == "tier_1" for edu in candidate.get("education", [])))
    
    # 3. Behavioral Signals
    profile_completeness_score = signals.get("profile_completeness_score", np.nan)
    profile_views_received_30d = signals.get("profile_views_received_30d", 0)
    applications_submitted_30d = signals.get("applications_submitted_30d", 0)
    
    recruiter_response_rate = signals.get("recruiter_response_rate", np.nan)
    avg_response_time_hours = signals.get("avg_response_time_hours", np.nan)
    connection_count = signals.get("connection_count", 0)
    endorsements_received = signals.get("endorsements_received", 0)
    
    notice_period_days = signals.get("notice_period_days", np.nan)
    willing_to_relocate = float(signals.get("willing_to_relocate", False))
    
    # GitHub Activity
    github_act = signals.get("github_activity_score", -1)
    github_missing = float(github_act == -1)
    github_activity_score = float(github_act) if github_act != -1 else np.nan
    
    # Offer Acceptance
    offer_acc = signals.get("offer_acceptance_rate", -1.0)
    offer_acceptance_missing = float(offer_acc == -1.0)
    offer_acceptance_rate = float(offer_acc) if offer_acc != -1.0 else np.nan
    
    # 4. Credibility & Honeypot indicators
    salary_min = signals.get("expected_salary_range_inr_lpa", {}).get("min", 0.0)
    salary_max = signals.get("expected_salary_range_inr_lpa", {}).get("max", 0.0)
    salary_max_lt_min = float(salary_max < salary_min)
    
    signup_d = parse_date(signals.get("signup_date"))
    active_d = parse_date(signals.get("last_active_date"))
    active_before_signup = float(signup_d is not None and active_d is not None and active_d < signup_d)
    
    total_skill_months = max([sk.get("duration_months", 0) for sk in skills]) if skills else 0
    skill_duration_gt_experience = float(total_skill_months > (years_exp * 12 + 6))
    
    total_job_months = sum(job_durations)
    years_exp_anomaly = float(total_job_months > (years_exp * 12 + 24))
    
    # Honeypot heuristic (M2 flag proxy - flags candidates with 3 or more simultaneous inconsistencies)
    m2_honeypot_flag = float(salary_max_lt_min + active_before_signup + skill_duration_gt_experience + years_exp_anomaly >= 3)
    
    return {
        "candidate_id": cid,
        "embedding_similarity": embedding_similarity,
        "retrieval_similarity": retrieval_similarity,
        "jd_required_skill_matches": jd_required_skill_matches,
        "jd_preferred_skill_matches": jd_preferred_skill_matches,
        "years_of_experience": years_exp,
        "avg_job_tenure_months": avg_job_tenure_months,
        "num_jobs": num_jobs,
        "short_tenure_count": short_tenure_count,
        "is_tier_1_education": is_tier_1_education,
        "profile_completeness_score": profile_completeness_score,
        "profile_views_received_30d": float(profile_views_received_30d),
        "applications_submitted_30d": float(applications_submitted_30d),
        "recruiter_response_rate": recruiter_response_rate,
        "avg_response_time_hours": avg_response_time_hours,
        "connection_count": float(connection_count),
        "endorsements_received": float(endorsements_received),
        "notice_period_days": notice_period_days,
        "willing_to_relocate": willing_to_relocate,
        "github_activity_score": github_activity_score,
        "github_missing": github_missing,
        "offer_acceptance_rate": offer_acceptance_rate,
        "offer_acceptance_missing": offer_acceptance_missing,
        "salary_max_lt_min": salary_max_lt_min,
        "active_before_signup": active_before_signup,
        "skill_duration_gt_experience": skill_duration_gt_experience,
        "years_exp_anomaly": years_exp_anomaly,
        "m2_honeypot_flag": m2_honeypot_flag
    }

def build_feature_matrix(candidates):
    """
    Transforms a list of candidates into a processed feature DataFrame.
    Implements imputations defined in configs/feature_schema.yaml.
    """
    records = [extract_candidate_features(c) for c in candidates]
    df = pd.DataFrame(records)
    
    # Impute missing values per feature_schema policies
    # fill_mean
    for col in ["embedding_similarity", "avg_job_tenure_months"]:
        if col in df.columns:
            mean_val = df[col].mean()
            if pd.isna(mean_val):
                mean_val = 0.5 if col == "embedding_similarity" else 24.0
            df[col] = df[col].fillna(mean_val)
            
    # fill_zero
    for col in ["retrieval_similarity", "jd_required_skill_matches", "jd_preferred_skill_matches", "github_activity_score"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
            
    # fill_median
    for col in ["profile_completeness_score", "recruiter_response_rate", "avg_response_time_hours", "notice_period_days", "offer_acceptance_rate"]:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                # Default clean values if subset has no valid
                defaults = {
                    "profile_completeness_score": 50.0,
                    "recruiter_response_rate": 0.44,
                    "avg_response_time_hours": 130.0,
                    "notice_period_days": 90.0,
                    "offer_acceptance_rate": 0.48
                }
                median_val = defaults.get(col, 0.0)
            df[col] = df[col].fillna(median_val)
            
    return df
