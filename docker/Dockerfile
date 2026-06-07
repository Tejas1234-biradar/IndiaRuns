FROM python:3.11-slim
WORKDIR /app
COPY requirements_runtime.txt .
RUN pip install --no-cache-dir -r requirements_runtime.txt
COPY runtime_pipeline/ .
CMD ["python", "rank.py"]
