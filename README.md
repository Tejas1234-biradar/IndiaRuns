# IndiaRuns

AI-powered candidate ranking pipeline. Two-phase architecture: heavy offline pre-computation + fast constrained runtime inference.

## Repository structure
See `docs/architecture.md` for full data flow documentation.

## Phase 1: Offline pipeline
Run on your own machine with GPU/internet access. Produces three artifacts:
- `artifacts/faiss_index.bin` — semantic vector index of all candidates
- `artifacts/model.xgb` — trained XGBoost ranker
- `artifacts/honeypot_ids.pkl` — flagged candidate IDs

## Phase 2: Runtime pipeline
Runs inside a Docker sandbox. No internet, no GPU, must finish in under 5 minutes.
Entrypoint: `runtime_pipeline/rank.py`

## Interactive sandbox
A Streamlit sandbox is available at `streamlit_app.py` for manual review/demo flows using an input slice of up to 100 candidates from the original artifact pool.

Local launch:
```bash
streamlit run streamlit_app.py
```

Deployment notes:
- Streamlit Cloud / app file: `streamlit_app.py`
- deployment guide: `docs/SANDBOX_DEPLOYMENT.md`
- bundled sample input: `data/sample_candidate_ids_100.csv`

### Docker workflow
Build the runtime image from the repository root:

```bash
docker build -t indiaruns-runtime .
```

Run a single container and write the submission to a host-mounted output directory:

```bash
docker run --rm \
  -v /absolute/path/to/output:/output \
  indiaruns-runtime
```

This generates `/absolute/path/to/output/output.csv` and validates it before the container exits.

Optional environment overrides:

```bash
docker run --rm \
  -v /absolute/path/to/output:/output \
  -e OUTPUT_PATH=/output/team_123.csv \
  -e ARTIFACTS_DIR=/app/artifacts \
  -e CANDIDATES_PATH=data/candidates.jsonl \
  indiaruns-runtime
```

## Setup

Offline environment:
```bash
pip install -r requirements_offline.txt
```

Runtime environment:
```bash
pip install -r requirements_runtime.txt
```

## Changelog
See `docs/CHANGELOG.md` for a full history of changes and decisions.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md)

## License
MIT
