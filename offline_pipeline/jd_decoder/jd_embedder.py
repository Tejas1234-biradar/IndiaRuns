import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

JD_CONFIG_PATH = "../../artifacts/jd_structured_config(2.5).json"
JD_VECTOR_OUTPUT_PATH = "../../artifacts/jd_query_vector.npy"

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

if __name__ == "__main__":
    query = load_and_prepare_query()
    vector = encode_jd_query(query)
