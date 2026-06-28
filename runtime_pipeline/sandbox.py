from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from runtime_pipeline.rank import (
    _resolve,
    attach_reasoning,
    load_honeypots,
    load_ranker,
    load_selected_features_parquet,
    normalize_scores,
    refresh_faiss_column,
    score_candidates,
    select_top_k,
)

MAX_SANDBOX_CANDIDATES = 100
SUPPORTED_SUFFIXES = (".csv", ".txt", ".jsonl", ".jsonl.gz")


def _dedupe_preserve_order(candidate_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for cid in candidate_ids:
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def parse_candidate_ids_from_upload(filename: str, payload: bytes) -> list[str]:
    lower = filename.lower()
    if lower.endswith(".csv"):
        text = payload.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if row]
        if not rows:
            return []
        header = [cell.strip().lower() for cell in rows[0]]
        if "candidate_id" in header:
            idx = header.index("candidate_id")
            ids = [
                row[idx].strip()
                for row in rows[1:]
                if len(row) > idx and row[idx].strip()
            ]
        else:
            ids = [row[0].strip() for row in rows if row[0].strip()]
        return _dedupe_preserve_order(ids)

    if lower.endswith(".txt"):
        text = payload.decode("utf-8")
        ids = [line.strip() for line in text.splitlines() if line.strip()]
        return _dedupe_preserve_order(ids)

    raw = gzip.decompress(payload) if lower.endswith(".gz") else payload
    text = raw.decode("utf-8")
    ids: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cid = str(obj.get("candidate_id", "")).strip()
        if cid:
            ids.append(cid)
    return _dedupe_preserve_order(ids)


def load_candidate_ids_from_path(path: str | Path) -> list[str]:
    p = Path(path)
    return parse_candidate_ids_from_upload(p.name, p.read_bytes())


def compute_slice_similarity(
    candidate_ids: list[str],
    index_path: Path,
    jd_vector_path: Path,
    candidate_ids_path: Path,
) -> tuple[dict[str, float], list[str]]:
    with candidate_ids_path.open(encoding="utf-8") as fh:
        all_ids: list[str] = json.load(fh)
    id_to_faiss = {cid: i for i, cid in enumerate(all_ids)}

    jd_vector = np.load(jd_vector_path).astype(np.float32).reshape(-1)
    index = faiss.read_index(str(index_path))

    similarity_map: dict[str, float] = {}
    missing_ids: list[str] = []
    for cid in candidate_ids:
        faiss_id = id_to_faiss.get(cid)
        if faiss_id is None:
            missing_ids.append(cid)
            continue
        vec = index.reconstruct(int(faiss_id))
        similarity_map[cid] = float(np.dot(jd_vector, vec))
    return similarity_map, missing_ids


def rank_candidate_slice(
    candidate_ids: list[str],
    artifacts_dir: str = "artifacts",
) -> tuple[pd.DataFrame, dict]:
    candidate_ids = _dedupe_preserve_order(candidate_ids)
    if not candidate_ids:
        raise RuntimeError("No candidate IDs were provided")
    if len(candidate_ids) > MAX_SANDBOX_CANDIDATES:
        raise RuntimeError(
            f"Sandbox accepts at most {MAX_SANDBOX_CANDIDATES} candidates; got {len(candidate_ids)}"
        )

    art = Path(artifacts_dir)
    faiss_path = _resolve(
        str(art / "faiss_index.bin"), ["artifacts/artifacts/faiss_index.bin"]
    )
    jd_path = _resolve(str(art / "jd_query_vector.npy"), [])
    ids_path = _resolve(str(art / "candidate_ids.json"), [])
    model_path = _resolve(str(art / "model.xgb"), [])
    honeypot_path = _resolve(str(art / "honeypot_ids.pkl"), [])
    features_path = _resolve(
        str(art / "features.parquet"), [str(art / "candidate_features.parquet")]
    )

    for label, path in [
        ("XGBoost model", model_path),
        ("honeypot IDs", honeypot_path),
        ("features parquet", features_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required sandbox artifact missing — {label}: {path}"
            )

    has_live_faiss = faiss_path.exists() and jd_path.exists() and ids_path.exists()

    honeypots = load_honeypots(honeypot_path)
    if has_live_faiss:
        similarity_map, missing_ids = compute_slice_similarity(
            candidate_ids, faiss_path, jd_path, ids_path
        )
        df = load_selected_features_parquet(features_path, list(similarity_map.keys()))
        df = refresh_faiss_column(df, similarity_map)
        similarity_source = "live_faiss"
    else:
        df = load_selected_features_parquet(features_path, candidate_ids)
        found_ids = set(df["candidate_id"].tolist())
        missing_ids = [cid for cid in candidate_ids if cid not in found_ids]
        similarity_source = "precomputed_feature_store"

    honeypot_ids = sorted(
        set(df.loc[df["candidate_id"].isin(honeypots), "candidate_id"].tolist())
    )
    df = df[~df["candidate_id"].isin(honeypots)].copy()
    if df.empty:
        raise RuntimeError(
            "All uploaded candidates were filtered out or missing from artifacts"
        )

    ranker = load_ranker(model_path)
    scored = score_candidates(df, ranker)
    top = select_top_k(scored, k=min(MAX_SANDBOX_CANDIDATES, len(scored)))
    top = attach_reasoning(top, ranker)
    top = attach_reasoning(
        top, ranker, raw_candidates_path if raw_candidates_path.exists() else None
    )
    final_scores = normalize_scores(top["model_score"].to_numpy())

    result = top[["candidate_id", "rank", "reasoning"]].copy()
    result.insert(2, "score", final_scores)

    metadata = {
        "requested_candidates": len(candidate_ids),
        "ranked_candidates": len(result),
        "missing_candidates": missing_ids,
        "honeypot_candidates": honeypot_ids,
        "similarity_source": similarity_source,
    }
    return result, metadata
