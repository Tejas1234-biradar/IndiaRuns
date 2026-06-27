# Student Ranker Training Summary (Task 3.4)

Training pipeline for the XGBoost Student Ranker, distilled from LLM Teacher labels (Task 3.3) and engineered numerical features (Task 2.3).

## Pipeline

```text
artifacts/labeled_candidates.json
        │
        ▼
scripts/inspect_teacher_labels.py  ──► artifacts/teacher_labels_integrity.json
        │
        ▼
scripts/build_training_dataset.py  ──► artifacts/training_dataset.parquet
        │                              artifacts/training_dataset_report.json
        ▼
scripts/train_ranker.py            ──► artifacts/model.xgb
                                       artifacts/training_metrics.json
```

## 1. Teacher Label Integrity

| Metric | Value |
|--------|-------|
| Total samples | 3,000 |
| Zero `overall_score` | 568 (18.9%) |
| Hard-reject zeros | 568 |
| Failed API labels | 0 (0.0%) |
| Ambiguous zeros (no evidence) | 89 |
| Duplicate IDs | 0 |
| Missing values | 0 |
| Score range | 0.00 – 8.20 (mean 1.63) |
| **Integrity check** | **PASS** |

Zero-score entries are predominantly legitimate `hard_reject` labels from the Teacher, not API failure placeholders. No samples were excluded from training.

## 2. Training Dataset

| Metric | Value |
|--------|-------|
| Input labels | 3,000 |
| Failed labels excluded | 0 |
| Training samples (after join) | 3,000 |
| Unmatched candidate IDs | 0 |
| Feature columns | 12 |
| Target column | `overall_score` |
| Target range | 0.00 – 8.20 |

Features are the 12 numerical columns from `feature_schema.py` (experience, FAISS similarity, behavioral signals, activity, tenure). Text fields (`evidence`, `concerns`) are excluded from the training matrix.

## 3. Model Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost `XGBRanker` |
| Objective | `rank:ndcg` |
| Relevance target | `round(overall_score)` clipped to [0, 10] |
| Query groups | Pseudo-groups of 50 candidates |
| Train / val split | 80% / 20% (2,400 / 600) |
| Learning rate | 0.05 |
| Max depth | 5 |
| Estimators | 300 (early stopping at 69) |

## 4. Validation Metrics

| Metric | Value |
|--------|-------|
| Validation RMSE | 2.83 |
| Validation MAE | 2.58 |
| Validation Spearman ρ | 0.53 |
| **Model accuracy (±1.0 pt)** | **5.2%** |
| Accuracy within ±2.0 pts | 34.7% |
| Exact grade match | 21.5% |
| Rank correlation | 52.5% |
| Prediction range (val) | −1.77 – 1.55 |
| Model reload check | PASS |

Spearman correlation on the held-out set confirms the ranker learns meaningful relative ordering from teacher scores. Ranker outputs are unbounded scores used for sorting candidates at inference time.

**Model accuracy** is reported as the percentage of validation samples where the predicted score is within ±1.0 of the teacher's `overall_score`. Rank correlation (52.5%) measures how well the model preserves relative ordering — often more meaningful than pointwise accuracy for a ranker.

## 5. Artifacts

| File | Description |
|------|-------------|
| `artifacts/training_dataset.parquet` | Joined feature matrix + target |
| `artifacts/model.xgb` | Trained Student Ranker |
| `artifacts/training_metrics.json` | Evaluation metrics |
| `artifacts/teacher_labels_integrity.json` | Label integrity report |

## Usage

Run the full pipeline from the repository root:

```bash
python scripts/inspect_teacher_labels.py
python scripts/build_training_dataset.py
python scripts/train_ranker.py
```

Each step validates its inputs and aborts if integrity checks fail or required artifacts are missing.
