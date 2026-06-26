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

if __name__ == "__main__":
    similarity_map = calculate_faiss_metrics()
    raw_records = load_base_records()
    print(f"Loaded {len(raw_records)} records and {len(similarity_map)} FAISS scores.")