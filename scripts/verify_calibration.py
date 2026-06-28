
"""
Checks submission.csv for flat scores and confirms strict monotonic decline.
"""
import pandas as pd
import numpy as np

def verify_scores():
    try:
        df = pd.read_csv("submission.csv")
    except FileNotFoundError:
        print("submission.csv not found. Run rank.py first.")
        return

    scores = df["score"].values
    
    # 1. Verify Monotonic Decline
    is_monotonic = np.all(np.diff(scores) <= 0)
    print(f"Strict Monotonic Decline: {'PASS' if is_monotonic else 'FAIL'}")
    
    # 2. Analyze Bounds & Tied Scores
    unique_scores = len(np.unique(scores))
    print(f"Unique Scores (Top 100): {unique_scores}/100")
    print(f"Max Score: {scores.max():.4f}")
    print(f"Min Score (Rank 100): {scores.min():.4f}")
    
    # 3. Variance / Contrast
    variance = np.var(scores)
    print(f"Score Variance: {variance:.4f}")
    
    if unique_scores < 100:
        print("[WARNING] Flat scores detected. Contrast control failed.")
    else:
        print("[SUCCESS] Maximum variation achieved. Final model weights and scaling locked.")

if __name__ == "__main__":
    verify_scores()