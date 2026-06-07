# Changelog

All notable changes to IndiaRuns are documented here, most recent first.
Format: date, title, what changed, why, scope.

---

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
