import os
import json
import numpy as np
import pandas as pd
import faiss

# --- SYSTEM PATHS ---
PARSED_CANDIDATES_PATH = "../../artifacts/candidates_parsed.jsonl"
JD_VECTOR_PATH = "../../artifacts/jd_query_vector.npy"
INDEX_PATH = "../../artifacts/faiss_index.bin"
CANDIDATE_IDS_PATH = "../../artifacts/candidate_ids.json"

def calculate_faiss_metrics():
    """
    Integrate FAISS similarity metrics
    Retrieves the exact Cosine Similarity score for ALL 100,000 candidates against the JD.
    """
    print("Loading FAISS index and JD query vector...")
    index = faiss.read_index(INDEX_PATH)
    jd_vector = np.load(JD_VECTOR_PATH).reshape(1, -1)
    
    with open(CANDIDATE_IDS_PATH, 'r') as f:
        candidate_ids = json.load(f)
        
    # Search all N vectors to get a complete score mapping
    print(f"Calculating similarity for {index.ntotal} candidates...")
    distances, indices = index.search(jd_vector, k=index.ntotal)
    
    # Map back to UUIDs
    similarity_map = {}
    for rank, (faiss_id, score) in enumerate(zip(indices[0], distances[0])):
        similarity_map[candidate_ids[faiss_id]] = float(score)
        
    return similarity_map

def load_base_records():
    """
    Load normalized candidate records
    """
    print("Loading parsed candidate stream...")
    records = []
    with open(PARSED_CANDIDATES_PATH, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return records

# ... (keep existing imports and functions) ...
from feature_schema import FEATURE_SCHEMA

def build_feature_matrix(raw_records, similarity_map):
    """
    [x] Integrate behavioral signals
    [x] Engineer ranking features
    [x] Handle missing values
    """
    print("Engineering features based on M3 Schema...")
    processed_rows = []
    
    for rec in raw_records:
        cid = rec['candidate_id']
        row = {'candidate_id': cid}
        
        # 1. Base Features & Behavioral Signals
        row['years_of_experience'] = min(float(rec.get('years_of_experience', 0.0)), FEATURE_SCHEMA['years_of_experience']['clip_max'])
        row['num_previous_jobs'] = int(rec.get('num_jobs', 0))
        row['num_skills_listed'] = min(int(rec.get('num_skills', 0)), FEATURE_SCHEMA['num_skills_listed']['clip_max'])
        row['max_assessment_score'] = float(rec.get('max_assessment_score', 0.0))
        
        # M1 Integration
        row['faiss_distance_to_jd'] = similarity_map.get(cid, 0.0)
        
        # Behavioral Features with Sentinel (-1.0) handling
        row['recruiter_response_rate'] = float(rec.get('recruiter_response_rate', -1.0))
        row['interview_completion_rate'] = float(rec.get('interview_completion_rate', -1.0))
        row['github_activity_score'] = float(rec.get('github_activity_score', -1.0))
        row['profile_views_received_30d'] = int(rec.get('profile_views_received_30d', 0))
        
        # 2. Derived Tenure Features
        jobs = max(row['num_previous_jobs'], 1)
        row['avg_job_duration_months'] = (row['years_of_experience'] * 12) / jobs
        row['notice_period_days'] = int(rec.get('notice_period_days', 30)) # default imputation
        
        # 3. Missing Value Imputation (Replacing Sentinels with np.nan for Pandas handling)
        for col, val in row.items():
            if col in FEATURE_SCHEMA and 'sentinel_value' in FEATURE_SCHEMA[col]:
                if val == FEATURE_SCHEMA[col]['sentinel_value']:
                    row[col] = np.nan
                    
        processed_rows.append(row)
        
    return pd.DataFrame(processed_rows)

if __name__ == "__main__":
    similarity_map = calculate_faiss_metrics()
    raw_records = load_base_records()
    df = build_feature_matrix(raw_records, similarity_map)
    print(f"Matrix Assembled. Shape: {df.shape}")