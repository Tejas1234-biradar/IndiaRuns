import json
import os
import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer

JD_CONFIG_PATH = "../../artifacts/jd_structured_config(2.5).json"
JD_VECTOR_OUTPUT_PATH = "../../artifacts/jd_query_vector.npy"
INDEX_PATH = "../../artifacts/faiss_index.bin"
IDS_PATH = "../../artifacts/candidate_ids.json"

def load_and_prepare_query():
    print(f"Loading parsed JD configuration from {JD_CONFIG_PATH}...")

    if not os.path.exists(JD_CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file missing at {JD_CONFIG_PATH}.")
    
    with open(JD_CONFIG_PATH, 'r', encoding = 'utf-8') as f:
        jd_data = json.load(f)
    
    # Use synthetic ideal candidate string to minimize the asymmetric search gap
    query_string = jd_data.get("synthetic_ideal_candidate_embedding_string")

    if not query_string:
        raise KeyError("synthetic_ideal_candidate_embedding_string not found in parsed JSON.")
    
    print(f"Extracted Query Payload ({len(query_string)} characters).")
    return query_string

def encode_jd_query(query_string):
    print("Loading BAAI/bge-small-en-v1.5 model...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    # L2 normalization is required for the FlatIP index to act as Cosine Similarity.
    print("Encoding query string...")
    jd_vector = model.encode(query_string, normalize_embeddings=True)

    # Validation
    assert jd_vector.shape == (384,), f"Dimensionality Error: Expected (384,), got {jd_vector.shape}"
    print(f"Validation Passed: Vector dimensionality is exactly {jd_vector.shape[0]}.")

    d_vector = jd_vector.astype(np.float32)
    np.save(JD_VECTOR_OUTPUT_PATH, jd_vector)
    print(f"JD Vector serialized to {JD_VECTOR_OUTPUT_PATH}")
    
    return jd_vector

def benchmark_retrieval(jd_vector):
    index = faiss.read_index(INDEX_PATH)

    with open(IDS_PATH, 'r') as f:
        candidate_ids = json.load(f)

    print(f"Executing search against {index.ntotal} candidate vectors...")

    query_matrix = jd_vector.reshape(1, -1)
    
    start_time = time.time()
    distances, indices = index.search(query_matrix, k=10)
    search_duration = time.time() - start_time
    
    print(f"Similarity Calculation Speed: {search_duration:.5f} seconds")
    print(f"Constraints Check: Speed is {'PASS' if search_duration < 1.0 else 'FAIL'} (< 1.0s required)")

    print("\nTop 10 Candidate Retrievals: ")
    for rank, (faiss_id, score) in enumerate(zip(indices[0], distances[0])):
        actual_id = candidate_ids[faiss_id]
        print(f"Rank {rank+1:02d} | Candidate: {actual_id} | Cosine Similarity: {score:.4f}")

if __name__ == "__main__":
    query = load_and_prepare_query()
    vector = encode_jd_query(query)
    benchmark_retrieval(vector)
