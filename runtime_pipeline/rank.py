#!/usr/bin/env python3
"""
rank.py — runtime entrypoint

Constraints: no internet, no GPU, must complete in < 5 minutes.

Pipeline:
  1. FAISS semantic retrieval  (faiss_index.bin + jd_query_vector.npy)
  2. Feature assembly          (features.parquet or candidates.jsonl)
  3. XGBoost scoring           (model.xgb)
  4. Honeypot filtering        (honeypot_ids.pkl)
  5. Top-100 CSV submission
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import xgboost as xgb

from runtime_pipeline.utils.features import (
    FEATURE_COLUMNS,
    apply_imputation,
    build_features_from_raw,
)

try:
    import orjson

    def loads_json(line: str | bytes) -> dict:
        return orjson.loads(line)

except ImportError:
    def loads_json(line: str | bytes) -> dict:
        return json.loads(line)

CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")
TOP_K = 100


def _resolve(path: str, fallbacks: list[str]) -> Path:
    """Return first existing path among primary and fallback locations."""
    candidates = [Path(path), *[Path(p) for p in fallbacks]]
    for p in candidates:
        if p.exists():
            return p
    return Path(path)


def load_honeypots(path: Path) -> set[str]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_faiss_similarity(
    index_path: Path,
    jd_vector_path: Path,
    candidate_ids_path: Path,
    top_k: int = 2000,
) -> dict[str, float]:
    """Compute JD cosine similarity for top-K indexed candidates."""
    import time
    
    t0 = time.perf_counter()

    try:
        print(f"[FAISS] Loading index from {index_path} …")
        index = faiss.read_index(str(index_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index. File may be corrupted. Error: {e}")

    try:
        jd_vector = np.load(jd_vector_path).astype(np.float32).reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"Failed to load JD vector. Error: {e}")

    try:
        with candidate_ids_path.open(encoding="utf-8") as fh:
            candidate_ids: list[str] = json.load(fh)
    except Exception as e:
        raise RuntimeError(f"Failed to load candidate IDs mapping. Error: {e}")

    # Prevent out-of-bounds requests if index has fewer than 2000 vectors
    k = min(top_k, index.ntotal)
    print(f"[FAISS] Searching {index.ntotal:,} vectors for top {k} …")
    
    try:
        distances, indices = index.search(jd_vector, k=k)
    except Exception as e:
        raise RuntimeError(f"FAISS search execution failed. Error: {e}")

    similarity_map = {}
    for faiss_id, score in zip(indices[0], distances[0]):
        # Boundary validation: verify retrieved index IDs map cleanly
        if faiss_id < 0 or faiss_id >= len(candidate_ids):
            print(f"  [WARN] FAISS returned out-of-bounds ID: {faiss_id}")
            continue
        
        cid = candidate_ids[int(faiss_id)]
        similarity_map[cid] = float(score)

    elapsed = time.perf_counter() - t0
    print(f"[FAISS] Top-{k} retrieval completed in {elapsed:.4f}s")
    
    # Benchmark warning if we miss the target window
    if elapsed > 2.0:
        print(f"  [WARN] FAISS search exceeded 2.0s target envelope ({elapsed:.2f}s)")

    return similarity_map


def load_features_parquet(path: Path) -> pd.DataFrame:
    print(f"[LOAD] Reading precomputed features from {path} …")
    df = pd.read_parquet(path)
    print(f"  {len(df):,} candidates × {len(df.columns)} columns")
    return df


def refresh_faiss_column(df: pd.DataFrame, similarity_map: dict[str, float]) -> pd.DataFrame:
    """Overwrite faiss_distance_to_jd with live FAISS scores."""
    out = df.copy()
    out["faiss_distance_to_jd"] = out["candidate_id"].map(similarity_map)
    missing = int(out["faiss_distance_to_jd"].isna().sum())
    if missing:
        mean_score = float(out["faiss_distance_to_jd"].mean())
        out["faiss_distance_to_jd"] = out["faiss_distance_to_jd"].fillna(mean_score)
        print(f"  [WARN] {missing} candidates missing FAISS scores; filled with mean")
    return out


def build_features_from_jsonl(
    candidates_path: Path,
    similarity_map: dict[str, float],
    metadata: dict | None,
) -> pd.DataFrame:
    """Stream candidates.jsonl and build the feature matrix."""
    print(f"[PARSE] Streaming candidates from {candidates_path} …")
    open_fn = gzip.open if candidates_path.suffix.lower() == ".gz" else open
    mode = "rt" if candidates_path.suffix.lower() == ".gz" else "r"

    rows: list[dict] = []
    with open_fn(candidates_path, mode, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw = loads_json(line)
            except Exception:
                continue
            cid = raw.get("candidate_id")
            if not cid or not CANDIDATE_ID_PATTERN.match(cid):
                continue
            
            # --- NEW FILTER ---
            if cid not in similarity_map:
                continue
            # ------------------

            faiss_score = similarity_map.get(cid, 0.0)
            feat = build_features_from_raw(raw, faiss_score)
            feat["candidate_id"] = cid
            rows.append(feat)

    if not rows:
        raise RuntimeError(f"No valid candidates parsed from {candidates_path}")

    imputed = apply_imputation(rows, metadata)
    df = pd.DataFrame(imputed)
    print(f"  Parsed {len(df):,} candidates")
    return df


def load_ranker(model_path: Path) -> xgb.XGBRanker:
    print(f"[MODEL] Loading ranker from {model_path} …")
    ranker = xgb.XGBRanker()
    ranker.load_model(str(model_path))
    return ranker


def score_candidates(df: pd.DataFrame, ranker: xgb.XGBRanker) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    df = df.copy()
    df["model_score"] = ranker.predict(X)
    return df


def select_top_k(
    df: pd.DataFrame,
    honeypots: set[str],
    k: int = TOP_K,
) -> pd.DataFrame:
    pool = df[~df["candidate_id"].isin(honeypots)].copy()
    pool = pool.sort_values(
        ["model_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    top = pool.head(k).reset_index(drop=True)
    top["rank"] = top.index + 1
    return top


def build_reasoning(row: pd.Series) -> str:
    parts = [
        f"semantic fit {row['faiss_distance_to_jd']:.3f}",
        f"{row['years_of_experience']:.1f}y exp",
        f"assessment {row['max_assessment_score']:.0f}",
    ]
    return "Ranked by XGBoost student ranker: " + ", ".join(parts) + "."


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Ensure scores strictly decrease (submission validator requirement)."""
    out = []
    prev = None
    for s in scores:
        val = round(float(s), 4)
        if prev is not None and val >= prev:
            val = round(prev - 0.0001, 4)
        out.append(val)
        prev = val
    return np.array(out)


def write_submission(top: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores = normalize_scores(top["model_score"].to_numpy())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (_, row) in enumerate(top.iterrows()):
            writer.writerow([
                row["candidate_id"],
                int(row["rank"]),
                float(scores[i]),
                build_reasoning(row),
            ])
    print(f"[OUT] Wrote {len(top)} rows to {out_path}")


def rank_candidates(
    candidates_path: str,
    out_path: str,
    artifacts_dir: str = "artifacts",
) -> pd.DataFrame:
    """End-to-end ranking pipeline. Returns the top-K dataframe."""
    t0 = time.perf_counter()
    art = Path(artifacts_dir)

    faiss_path = _resolve(
        str(art / "faiss_index.bin"),
        ["artifacts/artifacts/faiss_index.bin"],
    )
    jd_path = _resolve(str(art / "jd_query_vector.npy"), [])
    ids_path = _resolve(str(art / "candidate_ids.json"), [])
    model_path = _resolve(str(art / "model.xgb"), [])
    honeypot_path = _resolve(str(art / "honeypot_ids.pkl"), [])
    features_path = _resolve(str(art / "features.parquet"), [str(art / "candidate_features.parquet")])
    metadata_path = _resolve(str(art / "feature_metadata.json"), [])

    for label, path in [
        ("FAISS index", faiss_path),
        ("XGBoost model", model_path),
        ("honeypot IDs", honeypot_path),
        ("candidate IDs", ids_path),
    ]:
        if not path.exists():
            print(f"Error: required artifact missing — {label}: {path}", file=sys.stderr)
            sys.exit(1)

    metadata = None
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as fh:
            metadata = json.load(fh)

    honeypots = load_honeypots(honeypot_path)
    print(f"[LOAD] {len(honeypots)} honeypot IDs loaded")

    if not jd_path.exists():
        print(
            "[WARN] jd_query_vector.npy not found — using faiss scores from features.parquet",
            file=sys.stderr,
        )
        if not features_path.exists():
            print(
                "Error: need jd_query_vector.npy or features.parquet for FAISS scores",
                file=sys.stderr,
            )
            sys.exit(1)
        df = load_features_parquet(features_path)
        similarity_map = dict(zip(df["candidate_id"], df["faiss_distance_to_jd"]))
    else:
        # Inside rank_candidates, replace the `else:` block logic for parquet loading:
        similarity_map = load_faiss_similarity(faiss_path, jd_path, ids_path, top_k=2000)
        if features_path.exists():
            df = load_features_parquet(features_path)
            # --- NEW FILTER: Shrink DataFrame to only top 2000 ---
            df = df[df["candidate_id"].isin(similarity_map.keys())].copy()
            df = refresh_faiss_column(df, similarity_map)
        else:
            df = build_features_from_jsonl(
                Path(candidates_path), similarity_map, metadata
            )

    ranker = load_ranker(model_path)
    scored = score_candidates(df, ranker)
    top = select_top_k(scored, honeypots, k=TOP_K)

    elapsed = time.perf_counter() - t0
    print(f"\n[DONE] Ranked top {TOP_K} in {elapsed:.1f}s")
    return top


def main() -> None:
    parser = argparse.ArgumentParser(description="India Runs candidate ranking runtime")
    parser.add_argument(
        "--candidates",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl (used when features.parquet is absent)",
    )
    parser.add_argument("--out", default="submission.csv", help="Output submission CSV")
    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Directory containing model.xgb, faiss_index.bin, etc.",
    )
    args = parser.parse_args()

    top = rank_candidates(args.candidates, args.out, args.artifacts)
    write_submission(top, Path(args.out))
    print("Ranking completed successfully.")


if __name__ == "__main__":
    main()
