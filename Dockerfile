FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_runtime.txt .
RUN pip install --no-cache-dir -r requirements_runtime.txt

# Copy the runtime execution files
COPY rank.py .
COPY runtime_pipeline/ ./runtime_pipeline/
COPY tests/ ./tests/

# Default entrypoint
ENTRYPOINT ["python", "rank.py"]
