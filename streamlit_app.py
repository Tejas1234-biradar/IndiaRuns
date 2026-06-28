from __future__ import annotations

from pathlib import Path

import streamlit as st

from runtime_pipeline.sandbox import (
    MAX_SANDBOX_CANDIDATES,
    load_candidate_ids_from_path,
    parse_candidate_ids_from_upload,
    rank_candidate_slice,
)

st.set_page_config(page_title="IndiaRuns Sandbox", layout="wide")

st.title("IndiaRuns Interactive Sandbox")
st.caption(
    "Upload a small candidate slice from the original dataset and inspect end-to-end ranking, scoring, and SHAP-based reasoning."
)

sample_path = Path("data/sample_candidate_ids_100.csv")
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload candidate slice",
        type=["csv", "txt", "jsonl", "gz"],
        help="Accepted formats: CSV/TXT of candidate IDs, or JSONL/JSONL.GZ with candidate_id fields. Max 100 candidates.",
    )
    use_sample = st.checkbox("Use bundled 100-candidate sample", value=uploaded is None)
    run = st.button("Run sandbox ranking", type="primary")

st.markdown(
    f"This sandbox is configured for **≤ {MAX_SANDBOX_CANDIDATES} candidates** and uses the same runtime artifacts as the submission pipeline: **FAISS + XGBoost + SHAP**."
)

if run:
    candidate_ids: list[str] = []
    source_name = ""
    try:
        if uploaded is not None:
            candidate_ids = parse_candidate_ids_from_upload(
                uploaded.name, uploaded.getvalue()
            )
            source_name = uploaded.name
        elif use_sample and sample_path.exists():
            candidate_ids = load_candidate_ids_from_path(sample_path)
            source_name = sample_path.name
        else:
            st.error("Upload a candidate slice or enable the bundled sample.")
            st.stop()

        if not candidate_ids:
            st.error("No candidate IDs were found in the provided file.")
            st.stop()
        if len(candidate_ids) > MAX_SANDBOX_CANDIDATES:
            st.error(
                f"Received {len(candidate_ids)} candidate IDs. Please upload at most {MAX_SANDBOX_CANDIDATES}."
            )
            st.stop()

        with st.spinner("Running FAISS lookup, XGBoost scoring, and SHAP reasoning …"):
            result, metadata = rank_candidate_slice(
                candidate_ids, artifacts_dir="artifacts"
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Requested", metadata["requested_candidates"])
        c2.metric("Ranked", metadata["ranked_candidates"])
        c3.metric("Honeypots removed", len(metadata["honeypot_candidates"]))

        st.success(f"Completed sandbox run for `{source_name}`")
        if metadata.get("similarity_source") == "precomputed_feature_store":
            st.warning(
                "Running in lightweight cloud mode: live FAISS artifacts are not present, so the app is using precomputed `faiss_distance_to_jd` values from `features.parquet`."
            )
        if metadata["missing_candidates"]:
            st.warning(
                "Some candidate IDs were not found in the precomputed artifacts: "
                + ", ".join(metadata["missing_candidates"][:10])
                + (" …" if len(metadata["missing_candidates"]) > 10 else "")
            )
        if metadata["honeypot_candidates"]:
            st.info(
                "Filtered honeypot IDs: "
                + ", ".join(metadata["honeypot_candidates"][:10])
                + (" …" if len(metadata["honeypot_candidates"]) > 10 else "")
            )

        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Download ranked CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name="sandbox_ranked_candidates.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.exception(exc)
else:
    st.info(
        "Upload a candidate slice or use the bundled sample, then click **Run sandbox ranking**."
    )
