import os
import json
import random
from explain_model import load_explanation_configs, explain_candidate

# Dynamic text mapping dictionary for positive drivers
POS_TEXTS = {
    "embedding_similarity": [
        "Strong semantic alignment with Senior AI search and retrieval specifications.",
        "Profile shows deep overlap with the NLP and intelligence layer requirements."
    ],
    "retrieval_similarity": [
        "Demonstrated experience building vector search or hybrid retrieval infrastructure.",
        "Proven background shipping vector database systems at scale."
    ],
    "jd_required_skill_matches": [
        "Matches core required skills including embeddings, vector search, and Python.",
        "Possesses a strong foundation in required AI engineering tools."
    ],
    "jd_preferred_skill_matches": [
        "Features valuable nice-to-have capabilities like LLM fine-tuning or rankers.",
        "Includes nice-to-have exposure to advanced ranking or model fine-tuning."
    ],
    "years_of_experience": [
        "Solid professional experience ({val:.1f} years) matching the target seniority level.",
        "Demonstrates seasoned expertise with {val:.1f} years of professional engineering."
    ],
    "avg_job_tenure_months": [
        "Maintains strong career stability with an average tenure of {val:.1f} months per job.",
        "Demonstrates longevity with high average job duration."
    ],
    "is_tier_1_education": [
        "Prestige academic foundation with education from a Tier-1 institution.",
        "Academic credentials include degrees from Tier-1 universities."
    ],
    "profile_completeness_score": [
        "High profile completeness showing strong detail and documentation.",
        "Comprehensive and well-documented platform profile details."
    ],
    "profile_views_received_30d": [
        "High demand on the platform with strong recent recruiter interest.",
        "Significant traction on the platform with recruiters."
    ],
    "recruiter_response_rate": [
        "Excellent response rate of {val:.1f}% to recruiter communications.",
        "Highly engaged on the platform with prompt recruiter response habits."
    ],
    "connection_count": [
        "Highly connected professional profile with a strong network.",
        "Possesses a well-established professional connection base."
    ],
    "endorsements_received": [
        "Highly recommended profile with {val:.0f} endorsements from peers.",
        "Received substantial endorsements validation for core engineering skills."
    ],
    "notice_period_days": [
        "Highly available for immediate or rapid hire (notice: {val:.0f} days).",
        "Extremely short notice period of {val:.0f} days enables immediate onboarding."
    ],
    "willing_to_relocate": [
        "Open to location requirements and willing to relocate.",
        "Geographically flexible and ready for relocation if needed."
    ],
    "github_activity_score": [
        "Active GitHub presence showing solid recent coding contributions.",
        "Solid open-source coding footprint verified via GitHub activity."
    ],
    "offer_acceptance_rate": [
        "Proven track record of high offer acceptance rate ({val:.1f}%).",
        "Demonstrated serious intent in historical recruiting processes."
    ]
}

# Dynamic text mapping dictionary for concerns/negative drivers
NEG_TEXTS = {
    "short_tenure_count": [
        "Note: Career history shows multiple short tenures under 12 months.",
        "Potential job stability concerns given multiple short job tenures."
    ],
    "avg_response_time_hours": [
        "Note: Slow response latency to messages on the platform ({val:.1f} hours).",
        "Platform activity shows response lag time concerns."
    ],
    "notice_period_days": [
        "Note: Long notice period ({val:.0f} days) presents scheduling and buyout risks.",
        "Hiring timeline impacted by a long notice period of {val:.0f} days."
    ],
    "github_missing": [
        "Lacks a linked GitHub profile to verify open-source coding activity.",
        "Coding activity unverified due to missing GitHub integration."
    ],
    "offer_acceptance_missing": [
        "No prior offer acceptance history available on the platform.",
        "Historical offer feedback is currently unavailable."
    ],
    "salary_max_lt_min": [
        "Disqualified: Platform salary max is lower than min, suggesting a honeypot anomaly.",
        "Anomalous salary inputs detected on expected package min/max."
    ],
    "active_before_signup": [
        "Disqualified: Platform login records predate signup dates, marking a honeypot.",
        "Profile activity timeline contains serious inconsistencies."
    ],
    "skill_duration_gt_experience": [
        "Disqualified: Skill duration exceeds total professional experience.",
        "Skill experience timeline contradicts declared years of experience."
    ],
    "years_exp_anomaly": [
        "Disqualified: Cumulative job duration exceeds declared experience anomaly.",
        "Declared professional timeline contains structural discrepancies."
    ],
    "m2_honeypot_flag": [
        "Disqualified: Inconsistent platform signals indicating a honeypot profile.",
        "Profile rejected due to multiple credibility check violations."
    ],
    "embedding_similarity": [
        "Lower keyword alignment with founding team core RAG components.",
        "Adjacent skills only with limited core search-retrieval overlap."
    ],
    "jd_required_skill_matches": [
        "Lacks some of the required core machine learning and search skills.",
        "Required skill coverage is lower than preferred baseline."
    ]
}

def generate_reasoning(candidate, features_dict, means, importances, score=None):
    """
    Generates a context-aware reasoning statement for a candidate based on model drivers.
    """
    pos_drivers, neg_drivers = explain_candidate(features_dict, means, importances)
    
    # Determine if candidate is rejected (either via score threshold or severe inconsistencies)
    is_rejected = False
    if score is not None and score < 2.5:
        is_rejected = True
    elif features_dict.get("m2_honeypot_flag", 0) > 0 or features_dict.get("active_before_signup", 0) > 0:
        is_rejected = True
        
    if is_rejected:
        # Generate disqualified statement
        for feat in ["salary_max_lt_min", "active_before_signup", "m2_honeypot_flag", "skill_duration_gt_experience", "years_exp_anomaly"]:
            if features_dict.get(feat, 0) > 0:
                templates = NEG_TEXTS.get(feat, ["Disqualified: Profile credibility checks failed."])
                # Seed deterministic selection based on candidate id to prevent random outputs during validation runs
                seed = sum(ord(char) for char in candidate.get("candidate_id", "CAND_0000000"))
                return templates[seed % len(templates)]
        return "Disqualified: Candidate signals violate strict platform consistency guidelines."
        
    pos_sentences = []
    # Get top 2 positive drivers
    for feat, score, val in pos_drivers[:2]:
        if feat in POS_TEXTS:
            templates = POS_TEXTS[feat]
            seed = sum(ord(char) for char in candidate.get("candidate_id", "")) + int(score * 100)
            tmpl = templates[seed % len(templates)]
            # Inject facts dynamically
            if feat == "years_of_experience":
                pos_sentences.append(tmpl.format(val=val))
            elif feat == "avg_job_tenure_months":
                pos_sentences.append(tmpl.format(val=val))
            elif feat == "recruiter_response_rate":
                pos_sentences.append(tmpl.format(val=val * 100))
            elif feat == "notice_period_days":
                pos_sentences.append(tmpl.format(val=val))
            elif feat == "endorsements_received":
                pos_sentences.append(tmpl.format(val=val))
            elif feat == "offer_acceptance_rate":
                pos_sentences.append(tmpl.format(val=val * 100))
            else:
                pos_sentences.append(tmpl)
                
    neg_sentence = ""
    # Get top 1 concern
    if neg_drivers:
        feat, score, val = neg_drivers[0]
        if feat in NEG_TEXTS:
            templates = NEG_TEXTS[feat]
            seed = sum(ord(char) for char in candidate.get("candidate_id", "")) + int(abs(score) * 100)
            tmpl = templates[seed % len(templates)]
            if feat == "notice_period_days":
                neg_sentence = tmpl.format(val=val)
            elif feat == "avg_response_time_hours":
                neg_sentence = tmpl.format(val=val)
            else:
                neg_sentence = tmpl
                
    # Combine sentences into a robust description
    if pos_sentences:
        main_text = " ".join(pos_sentences)
        if neg_sentence:
            return f"{main_text} {neg_sentence}"
        return main_text
    else:
        # Fallback text if no positive drivers
        yexp = candidate.get("profile", {}).get("years_of_experience", 0)
        return f"AI engineer with {yexp:.1f} years of experience; meets basic technical criteria; response rates are moderate."
