# Changelog

All notable changes to IndiaRuns are documented here, most recent first.
Format: date, title, what changed, why, scope.

---
## [2026-06-26] — implement deterministic Pydantic schema for Teacher LLM

### What changed
- Updated `teacher_prompt.py` to include `TeacherEvaluationSchema` (Pydantic).


### Why
To train the student XGBoost model, we require mathematically strictly bounded targets (0.0 to 10.0). By defining a Pydantic schema, we can force the Gemini API to return guaranteed structured JSON, preventing pipeline crashes caused by hallucinated or misformatted text responses. Injecting the parsed JD ensures the LLM evaluates against the dynamic criteria defined in Task 1.1.

### Scope
teacher-llm

## [2026-06-26] — add diverse teacher sampling pipeline (Task 2.4)

### What changed
- Created offline_pipeline/teacher_student/sample_teacher_candidates.py
- Added unit tests in tests/test_teacher_sampling.py

### Why
Built a stratified sampling pipeline to select 3,000 candidates for LLM Teacher
evaluation from the unified feature matrix (Task 2.3 output). Excludes all 65
confirmed honeypots before sampling. Classifies the honeypot-free pool of
99,935 candidates into 5 dynamically-thresholded quality segments — strong fit,
average match, weak fit, hidden gem, and keyword stuffer — using FAISS semantic
distance, listed skill count, and validated assessment scores. Draws a fixed
allocation (900/900/600/300/300) across segments with automatic shortfall
redistribution if any segment pool is too small. Validates the final sample for
duplicate IDs, honeypot leakage, missing segments, and segment dominance before
export. Exports artifacts/teacher_sample.jsonl with full numeric features and
text context (headline, career history, skills, embedding text) for each
sampled candidate, ready for M3's LLM teacher prompting in Task 3.3.

### Scope
sampling

## [2026-06-26] — assembled final tabular feature matrix for all candidates

### What changed
- Created `offline_pipeline/feature_engineering/assemble_features.py`
- Implemented global `index.search` to map `faiss_distance_to_jd` for all 100K candidates.
- Implemented derived calculations (e.g., `avg_job_duration_months`).
- Exported the final 100K-row dataset to `artifacts/candidate_features.parquet`.

### Why
Assembled the fully numeric, model-ready ranking matrix. Downcasted datatypes to save memory for the strict sandbox limits. Chose the `.parquet` format for serialization because it performs drastically faster reads inside `rank.py` compared to `.csv` and strictly preserves schema data types, preventing runtime type-inference errors.

### Scope
feature-matrix

## [2026-06-26] — defined feature matrix schema

### What changed
- Created `offline_pipeline/feature_engineering/feature_schema.py`
- Defined experience, skill, behavioural, and activity metrics.
- Added `print_schema_documentation()` utility to generate markdown specs.

### Why
Defined unified tabular feature schema to enable assembly of the final tabular feature matrix for all 100,000 candidates.

### Scope
schema

## [2026-06-16] — benchmark CPU retrieval speed against FAISS candidate index

### What changed
- Added `benchmark_retrieval` testing function to `jd_embedder.py`
- Integrated `faiss.read_index` and evaluated top-10 retrieval timings

### Why
To fulfill the runtime sandbox constraints, we must empirically prove that computing similarity between the JD query and the 100K candidates is highly performant on a CPU. The benchmark verifies that a Top-10 exact inner-product search over 100K vectors completes in ~10-20m. 

### Scope
jd-embedder

## [2026-06-16] — extract and embed JD query payload 

### What changed
- Created `offline_pipeline/jd_decoder/jd_embedder.py`
- Implemented logic to read `jd_structured_config.json`
- Added `encode_jd_query` logic using `sentence-transformers`
- Enforced L2 normalization and float32 data typing
- Saved resulting array to `artifacts/jd_query_vector.npy`

### Why
Encoded the JD payload into the exact same semantic vector space as the candidate dataset. Explicitly applied `normalize_embeddings=True` to guarantee mathematical compatibility with the FAISS `IndexFlatIP` search architecture (enabling Cosine Similarity).

### Scope
jd-embedder

## [2026-06-16] — serialize index and add synthetic recall test

### What changed
- Added `faiss.write_index` serialization to `artifacts/faiss_index.bin`
- Implemented an exact-match self-retrieval test

### Why
Serialized the C++ object to a binary file to satisfy the strict runtime constraints. `rank.py` will use `faiss.read_index()` to instantly load this artifact into memory, avoiding the compute penalty of dynamically injecting 100K vectors at runtime and keeping execution well under the 5-minute sandbox limit. The synthetic recall test guarantees the data structures remain perfectly aligned before deployment.

### Scope
indexer

## [2026-06-16] — add vectors to FAISS and implement ID mapping

### What changed
- Created `offline_pipeline/semantic_indexer/build_index.py`
- Updated `build_index.py` to execute `index.add()`
- Implemented `map_faiss_to_candidate` function

### Why
Inner Product was selected because the embedder L2-normalizes the outputs, making IP mathematically equivalent to Cosine Similarity. A Flat index is used because 100K vectors easily fit in memory and a brute-force exact search guarantees 100% recall without the approximation loss of IVF or HNSW.
Loaded the 100K candidate vectors into the FAISS C++ backend. Implemented implicit positional mapping. The integer FAISS returns serves as the exact O(1) lookup index.

### Scope
indexer

## [2026-06-16] — implement candidate vectorizer 

### What changed
- Created `offline_pipeline/candidate_embeddings/candidate_embedder.py`
- Added `*.npy` to `.gitignore`
- Added `offline_pipeline/*.json` to `.gitignore`

### Why
Built the core embedding script to convert the 100K candidate dataset into a semantic vector space. Implemented a streaming reader to keep memory footprint extremely low while flattening complex candidate JSON structures into dense, natural-language payloads (Headline + Summary + Skills + Career History). Encodes the corpus into L2-normalized 384-dimensional vectors using `BAAI/bge-small-en-v1.5`. Batch size explicitly optimized for 16GB GPU VRAM. Serializes the aligned outputs directly to disk as `candidate_embeddings.npy` and `candidate_ids.json`.

### Scope
embedder


## [2026-06-15] — design JD parsing prompt workflow 

### What changed
- Created `offline_pipeline/jd_decoder/jd_decoder.py`
- Generated `jd_structured_config(gemini_flash_model_no.).json`

### Why
Defined the structured prompt engineering workflow for LLM-based Job Description (JD) parsing. The workflow establishes a system for extracting specific outputs:
1. **Target Requirements:** Mandatory technical and soft skills.
2. **Hidden Text Markers:** Keywords or signals indicating potential "culture fit" or "hidden" constraints.
3. **Anti-patterns:** Explicit exclusionary criteria to filter out unqualified profiles.
etc.
This ensures consistent, deterministic extraction of JD requirements, serving as the foundation for the retrieval engine in later stages.

### Scope
decoder

## [2026-06-13] — add honeypot detection scanner (Task 2.2)

### What changed
- Created offline_pipeline/feature_engineering/detect_honeypots.py
- Added unit tests in tests/test_honeypot.py

### Why
Built a rule-based scanner to identify the ~80 synthetic honeypot profiles
embedded in the 100K dataset. Applies 5 signal-violation rules: salary
inversion, low completeness with high advanced skills, salary inversion
compounded with offer history, impossible job durations, and zero-duration
expert skill clusters. Confirmed 65 honeypots at the 3-rule threshold.
12 additional candidate rules were probed and rejected with quantitative
justification — each fired on too many legitimate candidates to be
honeypot-exclusive. Exports artifacts/honeypot_ids.pkl as the blocklist
for rank.py — any ID in this set is hard-zeroed before scoring to prevent
disqualification (spec limit: honeypot rate <= 10% in top 100).

### Scope
honeypot

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
