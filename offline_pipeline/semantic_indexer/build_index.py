import numpy as np
import faiss
import time
import json

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

    return index, embeddings, candidate_ids

if __name__ == "__main__":
    build_index()
