"""
Compiles raw SHAP signals and raw numeric features into customized, 
non-templated natural language justifications for recruiters.
"""

import random

# Define a wide vocabulary matrix of descriptive grammar blocks
# Ensure explicit mentions of dynamic numerical values ({val})
GRAMMAR_MATRIX = {
    "Years of Experience": {
        "Positive Driver": [
            "an extensive background of {val} years",
            "a robust {val}-year professional tenure",
        ],
        "Negative Driver": [
            "a relatively brief timeline of {val} years",
            "limited practical exposure ({val} years)",
        ]
    },
    "Semantic Resume Match": {
        "Positive Driver": [
            "exceptional semantic alignment with the job description",
            "highly relevant core competencies matching the JD",
        ],
        "Negative Driver": [
            "low semantic overlap with the target role",
            "missing critical keywords from the job description",
        ]
    },
    "GitHub Activity Level": {
        "Positive Driver": [
            "a strong open-source footprint (score: {val})",
            "highly active code contributions (score: {val})",
        ],
        "Negative Driver": [
            "minimal visible GitHub activity (score: {val})",
            "a sparse open-source portfolio (score: {val})",
        ]
    },
    "Availability/Notice Period": {
        "Positive Driver": [
            "a highly favorable notice period of {val} days",
            "immediate availability ({val} days)",
        ],
        "Negative Driver": [
            "a restrictive {val}-day notice period",
            "delayed onboarding availability ({val} days)",
        ]
    },
    "Average Job Tenure": {
        "Positive Driver": [
            "demonstrated loyalty with {val} months average tenure",
            "stable career progression ({val} months/job)",
        ],
        "Negative Driver": [
            "a history of short stints ({val} months average)",
            "frequent job transitions averaging {val} months",
        ]
    },
    "Skills Listed": {
        "Positive Driver": [
            "a comprehensive technical stack of {val} skills",
            "broad technological proficiency ({val} listed skills)",
        ],
        "Negative Driver": [
            "a narrow recorded skill set ({val} skills)",
            "limited listed technical competencies ({val} skills)",
        ]
    }
}

# Map human-readable names back to dataframe column names for value extraction
REVERSE_MAPPING = {
    "Years of Experience": "years_of_experience",
    "Semantic Resume Match": "faiss_distance_to_jd",
    "GitHub Activity Level": "github_activity_score",
    "Availability/Notice Period": "notice_period_days",
    "Average Job Tenure": "avg_job_duration_months",
    "Skills Listed": "num_skills_listed"
}

def generate_justification(candidate_row: dict, top_drivers: list[dict]) -> str:
    phrases = []
    
    for driver in top_drivers:
        feature_name = driver["feature"]
        impact = driver["impact"]
        
        # Safely extract the exact numerical value from the candidate's raw row
        col_name = REVERSE_MAPPING.get(feature_name)
        raw_val = candidate_row.get(col_name, "N/A")
        
        # Format the number for readability (e.g., 2.0 -> 2)
        if isinstance(raw_val, float):
            raw_val = round(raw_val, 1) if raw_val % 1 != 0 else int(raw_val)
            
        # Fetch the grammar block and dynamically inject the value
        blocks = GRAMMAR_MATRIX.get(feature_name, {}).get(impact)
        
        if blocks:
            # Randomly select a phrasing block to increase string variance
            selected_block = random.choice(blocks)
            phrases.append(selected_block.format(val=raw_val))
        else:
            # Fallback syntax mapped specifically to the feature name (No generic "Good candidate" fallbacks)
            modifier = "strong" if impact == "Positive Driver" else "concerning"
            phrases.append(f"{modifier} metrics in {feature_name.lower()} ({raw_val})")

    # Syntactic Blending Rules
    if len(phrases) == 3:
        justification = f"Driven primarily by {phrases[0]}, coupled with {phrases[1]}, and {phrases[2]}."
    elif len(phrases) == 2:
        justification = f"Driven primarily by {phrases[0]}, as well as {phrases[1]}."
    elif len(phrases) == 1:
        justification = f"Driven primarily by {phrases[0]}."
    else:
        # Failsafe if SHAP returns nothing
        justification = "Candidate evaluation generated based on cumulative baseline metrics."

    # Capitalize first letter strictly
    return justification[0].upper() + justification[1:]

if __name__ == "__main__":
    import pandas as pd
    from shap_explainer import CandidateExplainer
    
    print("Initializing SHAP Explainer and Phrasing Engine Audit...")
    
    # Load dependencies
    explainer = CandidateExplainer("artifacts/model.xgb")
    df = pd.read_parquet("artifacts/training_dataset.parquet")
    
    # Sort by a target proxy to get a mix of good/bad candidates
    df = df.sort_values(by="years_of_experience", ascending=False).head(5)
    
    # Extract drivers via SHAP
    all_drivers = explainer.get_top_drivers(df)
    
    # Conduct manual string variance audits across sample outputs
    print("\n--- STRING VARIANCE AUDIT ---")
    for idx, (_, row) in enumerate(df.iterrows()):
        drivers = all_drivers[idx]
        row_dict = row.to_dict()
        
        justification = generate_justification(row_dict, drivers)
        print(f"Candidate {idx+1}:")
        print(f"  {justification}\n")