"""
Lightweight SHAP TreeExplainer for candidate feature importance.
Extracts top 3 drivers (positive/negative) for the final ranked top 100 candidates.
"""

import time
import xgboost as xgb
import numpy as np
import pandas as pd
import shap

# Build feature-to-human-readable metadata conversion mappings
FEATURE_MAPPING = {
    "experience_years": "Years of Experience",
    "github_score": "GitHub Activity Level",
    "skills_match_count": "Required Skills Match",
    "faiss_distance": "Semantic Resume Match",
    "tenure_avg_months": "Average Job Tenure",
    "promotions_count": "Career Velocity",
    "leadership_keywords": "Leadership Experience",
    "education_tier": "Education Pedigree",
    "system_design_score": "System Design Keywords",
    "production_ml_score": "Production ML Keywords",
    "oss_contributions": "Open Source Contributions",
    "notice_period_days": "Availability/Notice Period"
}

class CandidateExplainer:
    def __init__(self, model_path: str):
        """
        Checklist [x]: Initialize TreeExplainer module with model.xgb
        """
        self.model = xgb.XGBRanker()
        self.model.load_model(model_path)
        
        # Initialize lightweight TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)