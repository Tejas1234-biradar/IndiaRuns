import numpy as np
import faiss
import time
import json
import os

EMBEDDINGS_PATH = "../../artifacts/candidate_embeddings.npy"
IDS_PATH = "../../artifacts/candidate_ids.json"
INDEX_PATH = "../../artifacts/faiss_index.bin"

def map_faiss_to_candidate(faiss_indices, candidate_ids):
    """
    Order is preserved during embedding. 
    FAISS ID aligns with the array index of candidate_ids list.
    O(1) loookup.
    """
    return [candidate_ids[idx] for idx in faiss_indices]

def build_index():
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    start_time = time.time()

    # Load embeddings into RAM
    embeddings = np.load(EMBEDDINGS_PATH)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    print(f"Loaded {embeddings.shape[0]} vectors of dimenstion {dimension} in {time.time() - start_time:.2f}s.")

    # Initialize FAISS index using Inner Product (yields cosine similarity due to vectors being normalized)
    print("Initializing FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dimension)

    print("Adding vectors to FAISS index...")
    index.add(embeddings)

    assert index.ntotal ==embeddings.shape[0], "Vector count mismatch"
    print(f"Successfully indexed {index.ntotal} vectors in {time.time() - start_time:.2f}s.")

    # Load candidate IDs into memory for mapping
    with open(IDS_PATH, 'r') as f:
        candidate_ids = json.load(f)

    # Sanity test and recall
    print("\nRunning Sanity recall test")
    # Simulate a query by taking the 0th candidate's vector
    query_vector = embeddings[0].reshape(1, -1)

    start_search = time.time()
    distances, faiss_indices = index.search(query_vector, k=3)
    search_time = time.time() - start_search

    matched_candidates = map_faiss_to_candidate(faiss_indices[0], candidate_ids)

    print(f"Search executed in {search_time:.5f} seconds.")
    print(f"Top Match Candidate ID: {matched_candidates[0]}")
    print(f"Top Match Similarity Score: {distances[0][0]:.4f}")
    assert distances[0][0] > 0.99, "Recall test failed: Exact vector match should score ~1.0"

    # Serialize the FAISS index to disk
    print(f"\nSerializing FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    file_size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"FAISS index serialized to {INDEX_PATH} ({file_size_mb:.2f} MB) in {time.time() - start_time:.2f}s.")

    return index, embeddings, candidate_ids

if __name__ == "__main__":
    build_index()
