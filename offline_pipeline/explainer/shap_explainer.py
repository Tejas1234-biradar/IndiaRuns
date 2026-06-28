"""
Lightweight SHAP TreeExplainer for candidate feature importance.
Extracts top 3 drivers (positive/negative) for the final ranked top 100 candidates.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from offline_pipeline.feature_engineering.feature_schema import get_feature_columns

# Build feature-to-human-readable metadata conversion mappings
FEATURE_MAPPING = {
    "years_of_experience": "Years of Experience",
    "num_previous_jobs": "Previous Jobs Count",
    "faiss_distance_to_jd": "Semantic Resume Match",
    "num_skills_listed": "Skills Listed",
    "max_assessment_score": "Assessment Score",
    "recruiter_response_rate": "Recruiter Response Rate",
    "interview_completion_rate": "Interview Completion Rate",
    "github_activity_score": "GitHub Activity Level",
    "days_since_active": "Days Since Last Active",
    "profile_views_received_30d": "Profile Views (30d)",
    "avg_job_duration_months": "Average Job Tenure",
    "notice_period_days": "Availability/Notice Period",
}


class CandidateExplainer:
    def __init__(self, model_path: str):
        """
        Checklist [x]: Initialize TreeExplainer module with model.xgb
        """
        import json

        self.model = xgb.XGBRanker()
        self.model.load_model(model_path)

        # Extract the underlying C++ Booster object
        booster = self.model.get_booster()

        # --- THE CORRECT SHAP/XGBOOST MONKEY PATCH ---
        # shap.TreeExplainer calls booster.save_raw(raw_format="json") to parse the model.
        # We must intercept this specific byte stream to fix the base_score string formatting.
        original_save_raw = booster.save_raw

        def patched_save_raw(*args, **kwargs):
            # 1. Get the original bytearray from XGBoost
            raw_bytes = original_save_raw(*args, **kwargs)

            # 2. Only intercept if it's a JSON bytearray (starts with '{')
            if isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes.startswith(b"{"):
                try:
                    # Decode bytes to Python dict
                    model_dict = json.loads(raw_bytes.decode("utf-8"))

                    # Navigate down to base_score
                    base_score = (
                        model_dict.get("learner", {})
                        .get("learner_model_param", {})
                        .get("base_score")
                    )

                    # Strip the brackets if they exist
                    if (
                        isinstance(base_score, str)
                        and base_score.startswith("[")
                        and base_score.endswith("]")
                    ):
                        model_dict["learner"]["learner_model_param"]["base_score"] = (
                            base_score.strip("[]")
                        )

                    # Re-encode back to bytearray for SHAP
                    return bytearray(json.dumps(model_dict).encode("utf-8"))
                except Exception as e:
                    # Failsafe: return original bytes if parsing fails
                    pass

            return raw_bytes

        # Apply the patch to this specific instance
        booster.save_raw = patched_save_raw
        # ---------------------------------------------

        # Initialize TreeExplainer safely bypassing the SKLearn wrapper
        self.explainer = shap.TreeExplainer(booster)

    def get_top_drivers(self, candidates_df: pd.DataFrame) -> list[list[dict]]:
        """
        Compute local SHAP values for the final top 100 rows
        """
        feature_cols = get_feature_columns()
        missing = [col for col in feature_cols if col not in candidates_df.columns]
        if missing:
            raise ValueError(
                "Candidates dataframe is missing required feature columns for SHAP: "
                f"{missing}"
            )

        # Ensure we only pass numeric model feature columns to SHAP
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

                candidate_drivers.append(
                    {
                        "feature": FEATURE_MAPPING.get(
                            raw_feature_name, raw_feature_name
                        ),
                        "impact": direction,
                        "magnitude": round(abs(shap_val), 4),
                    }
                )

            all_drivers.append(candidate_drivers)

        return all_drivers


if __name__ == "__main__":
    # Benchmark explainer computation overhead
    print("Loading model and dataset for benchmarking...")

    # Load model and top 100 rows of training data
    test_explainer = CandidateExplainer("artifacts/model.xgb")
    df = pd.read_parquet("artifacts/training_dataset.parquet").head(100)
    print(df.columns.tolist())
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
        print(
            f" - {driver['impact']}: {driver['feature']} (Magnitude: {driver['magnitude']})"
        )
