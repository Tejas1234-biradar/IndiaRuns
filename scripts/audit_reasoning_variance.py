
"""
Computes pairwise text similarity arrays across explanations and checks for boilerplate text twins.
"""

import csv
from difflib import SequenceMatcher
from pathlib import Path

def calculate_similarity(str1: str, str2: str) -> float:
    return SequenceMatcher(None, str1, str2).ratio()

def run_audit():
    submission_path = Path("submission.csv")
    if not submission_path.exists():
        print("[ERROR] submission.csv not found! Run rank.py first.")
        return

    reasonings = []
    candidate_ids = []
    
    with submission_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidate_ids.append(row["candidate_id"])
            reasonings.append(row.get("reasoning", ""))

    if not reasonings or all(r == "" for r in reasonings):
        print("[FAIL] No text found in the 'reasoning' column.")
        return

    print(f"[AUDIT] Loaded {len(reasonings)} candidate justifications.")

    # 1. Check for AI Hallucinations, code flags, or unparsed schemas
    leakage_flags = ["nan", "none", "null", "undefined", "[object", "{", "}", "placeholder", "todo"]
    flagged_entries = 0
    
    for cid, text in zip(candidate_ids, reasonings):
        lowered = text.lower()
        if any(flag in lowered for flag in leakage_flags):
            print(f"  [ALERT] Code flag or structural leakage detected in {cid}: '{text}'")
            flagged_entries += 1

    # 2. Pairwise Similarity Check (Levenshtein-style matrix profile)
    twin_count = 0
    similarity_threshold = 0.82  # Anything above 82% similarity is a cookie-cutter pattern twin
    total_pairs = 0
    running_sim_sum = 0.0

    for i in range(len(reasonings)):
        for j in range(i + 1, len(reasonings)):
            sim = calculate_similarity(reasonings[i], reasonings[j])
            running_sim_sum += sim
            total_pairs += 1
            
            if sim >= similarity_threshold:
                twin_count += 1
                if twin_count <= 5:  # Print the first few structural twins found
                    print(f"  [TWIN MATCH >= {similarity_threshold*100}%] between {candidate_ids[i]} and {candidate_ids[j]}")
                    print(f"    Text 1: {reasonings[i]}")
                    print(f"    Text 2: {reasonings[j]}")

    avg_similarity = running_sim_sum / total_pairs if total_pairs > 0 else 0
    print("\n--- Linguistic Audit Results ---")
    print(f"Average Pairwise Text Similarity: {avg_similarity * 100:.2f}%")
    print(f"Total Cookie-Cutter Pattern Twins Found: {twin_count}")
    print(f"Leaked Code Flags / Hallucinations: {flagged_entries}")
    
    if twin_count > 0 or flagged_entries > 0:
        print("[STATUS] FAIL: Text variety is too low or contains structural code leaks.")
    else:
        print("[STATUS] PASS: Justifications sound diverse, professional, and unique.")

if __name__ == "__main__":
    run_audit()