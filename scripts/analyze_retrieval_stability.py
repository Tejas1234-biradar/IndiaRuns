"""
Simulates semantic shifts (synonyms, length variations) to test FAISS recall stability.
"""

import json
import time
import tracemalloc
from pathlib import Path
import faiss
import numpy as np

TOP_K = 2000
CRITICAL_TOP_N = 50 # We define the top 50 from the base JD as "critical targets"

def load_artifacts(art_dir: Path):
    index = faiss.read_index(str(art_dir / "faiss_index.bin"))
    jd_vector = np.load(art_dir / "jd_query_vector.npy").astype(np.float32).reshape(1, -1)
    with (art_dir / "candidate_ids.json").open("r", encoding="utf-8") as f:
        candidate_ids = json.load(f)
    return index, jd_vector, candidate_ids

def simulate_semantic_shift(base_vector: np.ndarray, variance: float, scale: float = 1.0) -> np.ndarray:
    """Simulates wording variants (noise) and sequence length changes (magnitude scaling)."""
    noise = np.random.normal(0, variance, base_vector.shape).astype(np.float32)
    return (base_vector + noise) * scale

def get_top_ids(index, vector, candidate_ids, k=TOP_K) -> set:
    _, indices = index.search(vector, k)
    return {candidate_ids[idx] for idx in indices[0] if 0 <= idx < len(candidate_ids)}

def main():
    art_dir = Path("artifacts")
    print("Loading artifacts...")
    index, base_vector, candidate_ids = load_artifacts(art_dir)
    
    # 1. Baseline Retrieval
    base_top_2000 = get_top_ids(index, base_vector, candidate_ids)
    
    # Extract "critical targets" (Top 50 from the baseline)
    _, base_indices = index.search(base_vector, CRITICAL_TOP_N)
    critical_targets = {candidate_ids[idx] for idx in base_indices[0]}
    print(f"Isolated {len(critical_targets)} critical target profiles.")

    # 2. Simulate Phrasing Synonyms & Intersection Variation
    print("\n--- Testing Semantic Shifts (Synonyms/Phrasing) ---")
    variants = [
        ("Minor Shift (Synonyms)", 0.05, 1.0),
        ("Moderate Shift (Rephrasing)", 0.10, 1.0),
        ("Sequence Length Var (Shortened)", 0.05, 0.8),
        ("Sequence Length Var (Lengthened)", 0.05, 1.2),
    ]

    for name, variance, scale in variants:
        shifted_vector = simulate_semantic_shift(base_vector, variance, scale)
        variant_top_2000 = get_top_ids(index, shifted_vector, candidate_ids)
        
        # Measure Intersection
        intersection = len(base_top_2000.intersection(variant_top_2000))
        overlap_pct = (intersection / TOP_K) * 100
        
        # Verify Critical Targets
        critical_retained = len(critical_targets.intersection(variant_top_2000))
        dropped = len(critical_targets) - critical_retained
        
        print(f"[{name}] Top-2000 Overlap: {overlap_pct:.1f}% | Critical Profiles Dropped: {dropped}")
        if dropped > 0:
            print(f"  [WARNING] Semantic boundary exceeded! {dropped} critical candidates lost.")

    # 3. Profile FAISS RAM Footprint under parallel/repeated loads
    print("\n--- Profiling FAISS RAM Footprint ---")
    tracemalloc.start()
    
    for _ in range(100):
        # Repeated queries simulating high load
        index.search(base_vector, TOP_K)
        
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak RAM usage during 100 queries: {peak / 10**6:.2f} MB")

if __name__ == "__main__":
    main()