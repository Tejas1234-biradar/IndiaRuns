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
