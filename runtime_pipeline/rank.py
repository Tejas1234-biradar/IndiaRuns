#!/usr/bin/env python3
"""
rank.py — runtime entrypoint
Constraints: no internet, no GPU, must complete in < 5 minutes
"""

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

# Try to use orjson for speed, fallback to stdlib json
try:
    import orjson as json_lib
except ImportError:
    import json as json_lib

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def mock_rank(candidates_path, out_path):
    print(f"Reading candidates from {candidates_path}...")
    path = Path(candidates_path)
    if not path.exists():
        print(f"Error: candidates file not found at {candidates_path}", file=sys.stderr)
        sys.exit(1)

    open_fn = gzip.open if path.suffix.lower() == ".gz" else open
    mode = "rt" if path.suffix.lower() == ".gz" else "r"

    valid_candidates = []
    
    with open_fn(path, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # orjson loads bytes or str, stdlib json loads str
                record = json_lib.loads(line)
                cid = record.get("candidate_id")
                if cid and CANDIDATE_ID_PATTERN.match(cid):
                    valid_candidates.append(cid)
                    if len(valid_candidates) >= 100:
                        break
            except Exception as e:
                # Ignore malformed lines during parsing
                continue

    if len(valid_candidates) < 100:
        print(f"Warning: Only found {len(valid_candidates)} valid candidates. Padding to 100.", file=sys.stderr)
        # Pad with mock IDs if needed (e.g. CAND_9000000 and up)
        while len(valid_candidates) < 100:
            mock_id = f"CAND_{9000000 + len(valid_candidates):07d}"
            if mock_id not in valid_candidates:
                valid_candidates.append(mock_id)

    # Output CSV columns: candidate_id, rank, score, reasoning
    # Scores must be non-increasing by rank
    # Ranks must be 1 to 100
    rows = []
    for i, cid in enumerate(valid_candidates):
        rank = i + 1
        score = round(1.0 - (i * 0.005), 4) # Decreasing scores: 1.0, 0.995, 0.990, ...
        reasoning = f"Ranked #{rank} based on mock semantic matching scores and credentials of candidate {cid}."
        rows.append([cid, rank, score, reasoning])

    print(f"Writing mock submission to {out_path}...")
    out_dir = Path(out_path).parent
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        writer.writerows(rows)

    print("Mock ranking completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="India Runs candidate ranking runtime script")
    parser.add_argument("--candidates", default="data/candidates.jsonl", help="Path to candidates.jsonl (or .gz)")
    parser.add_argument("--out", default="submission.csv", help="Path to output submission CSV")
    
    args = parser.parse_args()
    
    mock_rank(args.candidates, args.out)


if __name__ == "__main__":
    main()
