FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_runtime.txt .
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

# Optional: JD query vector for live FAISS scoring (generate offline via jd_embedder.py)
# COPY artifacts/jd_query_vector.npy ./artifacts/jd_query_vector.npy

# Default entrypoint
ENTRYPOINT ["python", "rank.py"]
