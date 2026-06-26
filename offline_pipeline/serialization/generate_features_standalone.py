"""
Standalone feature matrix generator for serialization.
Generates features without requiring FAISS index (uses mock similarity).
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering.feature_schema import FEATURE_SCHEMA

PARSED_CANDIDATES_PATH = "artifacts/candidates_parsed.jsonl"
CANDIDATE_IDS_PATH = "artifacts/candidate_ids.json"
OUTPUT_MATRIX_PATH = "artifacts/candidate_features.parquet"


def load_base_records(limit=None):
    """Load normalized candidate records."""
    print("Loading parsed candidate stream...")
    records = []
    with open(PARSED_CANDIDATES_PATH, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            records.append(json.loads(line))
    return records


def mock_faiss_similarity(candidate_id, index):
    """Generate mock FAISS similarity score."""
    # Use candidate ID hash for consistency
    hash_val = hash(candidate_id) % 1000
    # Similarity score between 0 and 1
    return float(hash_val) / 1000.0


def build_feature_matrix(raw_records):
    """Engineer ranking features from raw records."""
    print("Engineering features based on M3 Schema...")
    processed_rows = []
    
    for rec in raw_records:
        cid = rec['candidate_id']
        row = {'candidate_id': cid}
        
        # 1. Base Features & Behavioral Signals
        row['years_of_experience'] = min(
            float(rec.get('years_of_experience', 0.0)), 
            FEATURE_SCHEMA['years_of_experience']['clip_max']
        )
        row['num_previous_jobs'] = int(rec.get('num_jobs', 0))
        row['num_skills_listed'] = min(
            int(rec.get('num_skills', 0)), 
            FEATURE_SCHEMA['num_skills_listed']['clip_max']
        )
        
        # Map mean_assessment_score as proxy for max_assessment_score
        row['max_assessment_score'] = float(rec.get('mean_assessment_score', 0.0))
        
        # M1 Integration - Use mock similarity
        row['faiss_distance_to_jd'] = mock_faiss_similarity(cid, None)
        
        # Behavioral/Activity Features with Sentinel (-1.0) handling
        row['github_activity_score'] = float(rec.get('github_activity_score', -1.0))
        row['recruiter_response_rate'] = float(rec.get('recruiter_resp_rate', -1.0))
        row['interview_completion_rate'] = float(rec.get('interview_comp_rate', -1.0))
        row['profile_views_received_30d'] = int(rec.get('profile_views_30d', 0))
        
        # Add the missing days_inactive mapped to schema's days_since_active
        row['days_since_active'] = int(rec.get('days_inactive', 365))
        
        # 2. Derived Tenure Features
        jobs = max(row['num_previous_jobs'], 1)
        row['avg_job_duration_months'] = (row['years_of_experience'] * 12) / jobs
        row['notice_period_days'] = int(rec.get('notice_period_days', 30))
        
        # 3. Missing Value Imputation
        for col, val in row.items():
            if col in FEATURE_SCHEMA and 'sentinel_value' in FEATURE_SCHEMA[col]:
                if val == FEATURE_SCHEMA[col]['sentinel_value']:
                    row[col] = np.nan
                    
        processed_rows.append(row)
        
    return pd.DataFrame(processed_rows)


def finalize_and_export(df):
    """Generate final feature matrix and export."""
    print("Applying global statistical imputations...")
    
    # Apply Schema Imputations
    for col in df.columns:
        if col in FEATURE_SCHEMA:
            rule = FEATURE_SCHEMA[col].get('imputation')
            if rule == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif rule == "fill_zero":
                df[col] = df[col].fillna(0.0)
            elif rule == "fill_median":
                df[col] = df[col].fillna(df[col].median())
            elif rule == "max_penalty":
                df[col] = df[col].fillna(365)  # Max days inactive
                
    # Ensure memory optimization
    print("Downcasting float types for memory optimization...")
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    print(f"Exporting feature matrix to {OUTPUT_MATRIX_PATH} (Parquet Format)...")
    df.to_parquet(OUTPUT_MATRIX_PATH, engine='pyarrow', index=False)
    
    file_size_mb = os.path.getsize(OUTPUT_MATRIX_PATH) / (1024 * 1024)
    print(f"✓ Feature matrix serialized ({file_size_mb:.2f} MB).")
    
    return df


def main():
    try:
        raw_records = load_base_records()
        print(f"  ✓ Loaded {len(raw_records):,} candidates")
        
        df = build_feature_matrix(raw_records)
        df = finalize_and_export(df)
        
        print(f"\n✓ Feature matrix generated successfully!")
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)[:5]}...")
        
        return 0
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
