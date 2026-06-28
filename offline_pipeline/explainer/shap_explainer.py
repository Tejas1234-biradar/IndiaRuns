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
    
    def get_top_drivers(self, candidates_df: pd.DataFrame) -> list[list[dict]]:
        """
        Compute local SHAP values for the final top 100 rows
        """
        # Ensure we only pass numeric feature columns to SHAP
        feature_cols = list(FEATURE_MAPPING.keys())
        X_matrix = candidates_df[feature_cols]
        
        # Compute SHAP values (returns numpy array of shape [n_samples, n_features])
        shap_values = self.explainer.shap_values(X_matrix)
        
        all_drivers = []
        
        for i in range(len(candidates_df)):
            row_shap = shap_values[i]
            
            # Extract index locations of top 3 most influential features
            # argsort on absolute values to get the strongest drivers (both positive and negative)
            top_3_indices = np.argsort(np.abs(row_shap))[-3:][::-1]
            
            candidate_drivers = []
            for idx in top_3_indices:
                raw_feature_name = feature_cols[idx]
                shap_val = float(row_shap[idx])
                
                # Normalize feature contribution directions
                direction = "Positive Driver" if shap_val > 0 else "Negative Driver"
                
                candidate_drivers.append({
                    "feature": FEATURE_MAPPING.get(raw_feature_name, raw_feature_name),
                    "impact": direction,
                    "magnitude": round(abs(shap_val), 4)
                })
                
            all_drivers.append(candidate_drivers)
            
        return all_drivers

if __name__ == "__main__":
    # Benchmark explainer computation overhead
    print("Loading model and dataset for benchmarking...")
    
    # Load model and top 100 rows of training data
    test_explainer = CandidateExplainer("artifacts/model.xgb")
    df = pd.read_parquet("artifacts/training_dataset.parquet").head(100)
    
    print(f"Benchmarking SHAP extraction for {len(df)} candidates...")
    start_time = time.perf_counter()
    
    drivers = test_explainer.get_top_drivers(df)
    
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    
    print(f"Extraction completed in {elapsed_ms:.2f} ms")
    print(f"Average time per candidate: {elapsed_ms / len(df):.2f} ms")
    
    # Sanity check output
    print("\nSample Output for Top Candidate:")
    for driver in drivers[0]:
        print(f" - {driver['impact']}: {driver['feature']} (Magnitude: {driver['magnitude']})")