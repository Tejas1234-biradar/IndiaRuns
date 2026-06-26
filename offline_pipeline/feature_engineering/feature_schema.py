"""
Unified Tabular Feature Matrix Schema
Defines the exact columns, types, and imputation rules for the XGBoost Ranker.
"""

FEATURE_SCHEMA = {
    # --- [x] Experience Features ---
    "years_of_experience": {
        "type": "float",
        "description": "Total years of professional experience.",
        "imputation": "fill_zero", # If missing, assume 0
        "clip_max": 40.0           # Cap extreme outliers to 40 years
    },
    "num_previous_jobs": {
        "type": "int",
        "description": "Total count of jobs listed in career history.",
        "imputation": "fill_zero"
    },
    
    # --- [x] Skill Alignment Features ---
    "faiss_distance_to_jd": {
        "type": "float",
        "description": "Cosine similarity distance from M1 FAISS index.",
        "imputation": "mean", # Should not be missing, but fallback to mean
        "is_core_feature": True
    },
    "num_skills_listed": {
        "type": "int",
        "description": "Total count of skills on the profile.",
        "imputation": "fill_zero",
        "clip_max": 100
    },
    "max_assessment_score": {
        "type": "float",
        "description": "Highest score across all verified skill assessments.",
        "imputation": "fill_zero"
    },
    # ... (keep existing features) ...

    # --- [x] Behavioral Features ---
    "recruiter_response_rate": {
        "type": "float",
        "description": "Percentage of recruiter messages replied to (0.0 to 1.0).",
        "imputation": "mean", # If they haven't received messages, assume average behavior
        "sentinel_value": -1.0
    },
    "interview_completion_rate": {
        "type": "float",
        "description": "Percentage of accepted interviews actually attended.",
        "imputation": "fill_zero",
        "sentinel_value": -1.0
    },
    
    # --- [x] Activity Features ---
    "github_activity_score": {
        "type": "float",
        "description": "0-100 score of recent GitHub commits/PRs.",
        "imputation": "fill_zero", # Missing GitHub implies 0 activity
        "sentinel_value": -1.0
    },
    "days_since_active": {
        "type": "int",
        "description": "Days since last platform login.",
        "imputation": "max_penalty", # Missing means highly inactive (e.g., 365 days)
    },
    "profile_views_received_30d": {
        "type": "int",
        "description": "Inbound profile traffic.",
        "imputation": "fill_zero"
    },
}

def get_feature_columns():
    return list(FEATURE_SCHEMA.keys())