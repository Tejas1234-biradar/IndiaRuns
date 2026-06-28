# Interactive Sandbox Deployment

This repository now includes a Streamlit-compatible interactive sandbox entrypoint at `streamlit_app.py`.

## What it does

- accepts a candidate slice of **up to 100** records/IDs
- uses the same precomputed runtime artifacts as the submission pipeline
- runs **FAISS-based similarity refresh**, **XGBoost scoring**, and **SHAP-driven reasoning**
- displays ranked output in the browser and allows CSV download

Accepted upload formats:

- `.csv` with a `candidate_id` column (or candidate IDs in the first column)
- `.txt` with one candidate ID per line
- `.jsonl` / `.jsonl.gz` containing `candidate_id` fields

A bundled demo input is provided at `data/sample_candidate_ids_100.csv`.

## Local run

```bash
micromamba run -n IndiaRuns streamlit run streamlit_app.py
```

## Streamlit Cloud deployment

1. Push the repository to GitHub.
2. In Streamlit Cloud, create a new app from the repo.
3. Set the main file path to `streamlit_app.py`.
4. Ensure the repo includes the required artifacts under `artifacts/`.
5. Deploy.

The root `requirements.txt` installs both runtime dependencies and `streamlit`.

## Hugging Face Spaces deployment

If you prefer Hugging Face Spaces, use a **Docker Space** and point it at this repository.
A minimal approach is:

1. Create a new Docker Space.
2. Copy the repository contents into the Space.
3. Use `streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0` as the container start command.
4. Ensure the `artifacts/` directory is present in the Space storage.

## Verification target

Use the bundled sample or an uploaded slice of **≤100** original candidate IDs to verify end-to-end functionality.
