import json
import os
import pandas as pd
import numpy as np

# Define feature signs (1 for positive correlation with fit, -1 for negative correlation)
FEATURE_SIGNS = {
    "embedding_similarity": 1,
    "retrieval_similarity": 1,
    "jd_required_skill_matches": 1,
    "jd_preferred_skill_matches": 1,
    "years_of_experience": 1,
    "avg_job_tenure_months": 1,
    "num_jobs": 1,
    "is_tier_1_education": 1,
    "profile_completeness_score": 1,
    "profile_views_received_30d": 1,
    "applications_submitted_30d": 1,
    "recruiter_response_rate": 1,
    "connection_count": 1,
    "endorsements_received": 1,
    "willing_to_relocate": 1,
    "github_activity_score": 1,
    "offer_acceptance_rate": 1,
    
    # Negative features (higher value is worse)
    "short_tenure_count": -1,
    "avg_response_time_hours": -1,
    "notice_period_days": -1,
    "github_missing": -1,
    "offer_acceptance_missing": -1,
    "salary_max_lt_min": -1,
    "active_before_signup": -1,
    "skill_duration_gt_experience": -1,
    "years_exp_anomaly": -1,
    "m2_honeypot_flag": -1
}

def load_explanation_configs():
    """
    Loads baseline means and feature importances.
    """
    means_path = "models/feature_means.json"
    imp_path = "reports/feature_importance.csv"
    
    if not os.path.exists(means_path) or not os.path.exists(imp_path):
        raise FileNotFoundError("Model configurations not found. Run model training first.")
        
    with open(means_path, "r") as f:
        means = json.load(f)
        
    df_imp = pd.read_csv(imp_path)
    importances = dict(zip(df_imp["feature"], df_imp["importance"]))
    
    return means, importances

def explain_candidate(features_dict, means, importances):
    """
    Calculates feature contributions for a single candidate.
    Returns sorted lists of (feature_name, contribution_score) for positive and negative drivers.
    """
    contributions = []
    
    for feat, val in features_dict.items():
        if feat not in means or feat not in importances:
            continue
            
        mean_val = means[feat]
        imp_val = importances[feat]
        sign = FEATURE_SIGNS.get(feat, 1)
        
        # Contribution score represents deviation from baseline weighted by importance
        contr = sign * (val - mean_val) * imp_val
        contributions.append((feat, contr, val))
        
    # Sort contributions
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    # Positive drivers (contribution > 0)
    pos_drivers = [c for c in contributions if c[1] > 0]
    
    # Negative drivers / concerns (contribution < 0)
    neg_drivers = [c for c in contributions if c[1] < 0]
    neg_drivers.sort(key=lambda x: x[1]) # Most negative first
    
    return pos_drivers, neg_drivers

def main():
    # Test on a simulated candidate features dict
    print("Testing explainability module...")
    try:
        means, importances = load_explanation_configs()
        
        # Simulated strong candidate features
        test_features = {
            "embedding_similarity": 0.85,
            "retrieval_similarity": 0.75,
            "jd_required_skill_matches": 0.8,
            "jd_preferred_skill_matches": 0.6,
            "years_of_experience": 7.5,
            "avg_job_tenure_months": 36.0,
            "num_jobs": 2.0,
            "short_tenure_count": 0.0,
            "is_tier_1_education": 1.0,
            "profile_completeness_score": 90.0,
            "profile_views_received_30d": 80.0,
            "applications_submitted_30d": 4.0,
            "recruiter_response_rate": 0.85,
            "avg_response_time_hours": 24.0,
            "connection_count": 450.0,
            "endorsements_received": 65.0,
            "notice_period_days": 15.0,
            "willing_to_relocate": 1.0,
            "github_activity_score": 65.0,
            "github_missing": 0.0,
            "offer_acceptance_rate": 0.75,
            "offer_acceptance_missing": 0.0,
            "salary_max_lt_min": 0.0,
            "active_before_signup": 0.0,
            "skill_duration_gt_experience": 0.0,
            "years_exp_anomaly": 0.0,
            "m2_honeypot_flag": 0.0
        }
        
        pos, neg = explain_candidate(test_features, means, importances)
        
        print("\nTop 3 Positive Drivers:")
        for feat, score, val in pos[:3]:
            print(f"  {feat}: score={score:.4f}, value={val}")
            
        print("\nTop 3 Concerns / Negative Drivers:")
        for feat, score, val in neg[:3]:
            print(f"  {feat}: score={score:.4f}, value={val}")
            
    except Exception as e:
        print(f"Error testing explainer: {e}")

if __name__ == "__main__":
    main()
