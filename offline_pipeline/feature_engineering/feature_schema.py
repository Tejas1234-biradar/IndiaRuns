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
    }
}

def get_feature_columns():
    return list(FEATURE_SCHEMA.keys())