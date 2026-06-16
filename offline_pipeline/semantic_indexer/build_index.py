import numpy as np
import faiss
import time

EMBEDDINGS_PATH = "../../artifacts/candidate_embeddings.npy"
IDS_PATH = "../../artifacts/candidate_ids.json"
INDEX_PATH = "../../artifacts/faiss_index.bin"

def initialize_index():
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

    return index, embeddings

if __name__ == "__main__":
    initialize_index()
    