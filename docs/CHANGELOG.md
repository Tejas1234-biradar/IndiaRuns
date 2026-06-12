# Changelog

All notable changes to IndiaRuns are documented here, most recent first.
Format: date, title, what changed, why, scope.

---

## [2026-06-12] — add high-throughput candidate stream parser (Task 2.1)

### What changed
- Created offline_pipeline/feature_engineering/parse_candidates.py

### Why
Built a streaming parser for the 100K candidate dataset (candidates.jsonl) that never
loads the full file into memory. Uses orjson for parsing after benchmarking three
libraries (orjson 2.15× faster than stdlib json, ijson too slow for line-by-line JSONL).
Normalizes all nested candidate structures (profile, career_history, education, skills,
redrob_signals) into flat records and exports artifacts/candidates_parsed.jsonl for
consumption by M1 (embeddings) and M3 (feature matrix). Validated against all 50 known
sample candidates — 600 field checks, 0 errors. Peak memory 0.2 MB confirms the
streaming design holds under full dataset load.

### Scope
parser

## [2026-06-07] — write project README

### What changed
- Created `README.md` at the root of the repository

### Why
Created the project README file last so that all internal links and references to other components, requirements, and guides are fully valid and verifyable at the time of commit.

### Scope
docs

## [2026-06-07] — add architecture stub

### What changed
- Created `docs/architecture.md` placeholder

### Why
Created the architecture document stub. This will serve as a central reference to be filled in once the Phase 1 offline and Phase 2 runtime pipeline implementations are finalized.

### Scope
docs

## [2026-06-07] — add contributing guide

### What changed
- Created `CONTRIBUTING.md` guide

### Why
Created the contributing guide to document the atomic commit requirements and changelog maintenance rules. This ensures that all team members adhere to the same engineering workflow and repository standards from day one.

### Scope
ci

## [2026-06-07] — add PR template

### What changed
- Created `.github/` directory
- Added `.github/PULL_REQUEST_TEMPLATE.md`

### Why
Created the pull request template to enforce the hackathon's key constraints (preventing binary artifacts/data files from being committed, and ensuring rank.py runs correctly in the Docker sandbox) for all future contributions.

### Scope
ci

## [2026-06-07] — add offline and runtime requirements files

### What changed
- Created `requirements_offline.txt` containing dependencies for offline model training/indexing
- Created `requirements_runtime.txt` containing lightweight dependencies for runtime sandbox evaluation

### Why
Intentionally split dependencies into offline and runtime requirements. Offline dependencies (such as heavy API clients and deep learning libraries) cannot run in the resource-constrained Docker sandbox, whereas runtime dependencies are kept CPU-only and lightweight to comply with sandbox execution constraints.

### Scope
deps

## [2026-06-07] — add Dockerfile stub

### What changed
- Created `docker/` directory
- Added `docker/Dockerfile` using python:3.11-slim

### Why
Created the Dockerfile for containerization. It uses python:3.11-slim to keep the image size small for the sandbox environment, and intentionally only installs the runtime dependencies to optimize the build.

### Scope
docker

## [2026-06-07] — add rank.py entrypoint stub

### What changed
- Created `runtime_pipeline/rank.py` with runtime entrypoint structure

### Why
Created rank.py, which serves as the sole submission entrypoint for the candidate ranking pipeline. It must remain self-contained and run without external network calls or GPU access, satisfying the strict time limits of the evaluation environment.

### Scope
runtime

## [2026-06-07] — add placeholder test stubs

### What changed
- Created `tests/` directory
- Added test placeholder stubs: `test_honeypot.py`, `test_faiss_recall.py`, and `test_xgb_ranker.py`

### Why
Created placeholder test files for unit testing. These stubs will be filled in with actual test cases as each pipeline component is implemented.

### Scope
tests

## [2026-06-07] — scaffold pipeline and artifact directories

### What changed
- Created `offline_pipeline/` with subdirectories `jd_decoder/`, `semantic_indexer/`, `feature_engineering/`, and `teacher_student/`
- Created `runtime_pipeline/utils/` subdirectory
- Created `artifacts/` and `data/` directories with `.gitkeep` files

### Why
Scaffolded the initial directory structure. The four offline pipeline subdirectories map to the four phases of the offline pipeline (JD decode, embed, feature eng, train).

### Scope
Repo Infrastructure

## [2026-06-07] — add gitignore

### What changed
- Created `.gitignore` file to specify untracked files and directories

### Why
Committed gitignore first before any other files so that artifact binaries and raw data files are never accidentally tracked, which would bloat the repo and risk leaking sensitive candidate data.

### Scope
Repo Infrastructure

## [2026-06-07] — initialize repository structure

### What changed
- Created `docs/CHANGELOG.md` to track all repo changes with context

### Why
Changelog initialized as the first file so every subsequent change in this sprint has a documented record. Future teammates and agents can read this file to understand not just what exists in the repo but why decisions were made.

### Scope
Repo Infrastructure
