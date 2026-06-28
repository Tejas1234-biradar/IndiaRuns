FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARTIFACTS_DIR=/app/artifacts \
    CANDIDATES_PATH=data/candidates.jsonl \
    OUTPUT_PATH=/output/output.csv

COPY requirements_runtime.txt ./
RUN pip install --no-cache-dir -r requirements_runtime.txt

# Copy runtime code and pre-built artifacts
COPY rank.py .
COPY runtime_pipeline/ ./runtime_pipeline/
COPY tests/ ./tests/
COPY artifacts/model.xgb ./artifacts/model.xgb
COPY artifacts/faiss_index.bin ./artifacts/faiss_index.bin
COPY artifacts/candidate_ids.json ./artifacts/candidate_ids.json
COPY artifacts/honeypot_ids.pkl ./artifacts/honeypot_ids.pkl
COPY artifacts/features.parquet ./artifacts/features.parquet
COPY artifacts/feature_metadata.json ./artifacts/feature_metadata.json
COPY artifacts/jd_query_vector.npy ./artifacts/jd_query_vector.npy

RUN chmod +x /app/docker/entrypoint.sh && mkdir -p /output

ENTRYPOINT ["/app/docker/entrypoint.sh"]
