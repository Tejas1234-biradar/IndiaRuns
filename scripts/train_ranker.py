"""
scripts/train_ranker.py

Task 3.4 — Train the Student XGBoost Ranker on teacher labels.

Trains an XGBRanker on engineered numerical features with overall_score as
the relevance target. All candidates share a single query group (one JD),
so the model learns relative ordering within the labeled pool.

Usage:
    python scripts/train_ranker.py \\
        --training artifacts/training_dataset.parquet \\
        --model-out artifacts/model.xgb \\
        --metrics-out artifacts/training_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from offline_pipeline.feature_engineering.feature_schema import get_feature_columns


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank correlation without scipy dependency."""
    true_ranks = pd.Series(y_true).rank(method="average").to_numpy()
    pred_ranks = pd.Series(y_pred).rank(method="average").to_numpy()
    return float(np.corrcoef(true_ranks, pred_ranks)[0, 1])


def load_training_data(training_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load pre-built training parquet and validate columns."""
    path = Path(training_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {training_path}. "
            "Run scripts/build_training_dataset.py first."
        )

    df = pd.read_parquet(path)
    feature_cols = get_feature_columns()
    missing = [c for c in feature_cols + ["overall_score"] if c not in df.columns]
    if missing:
        raise ValueError(f"Training dataset missing columns: {missing}")

    if df[feature_cols + ["overall_score"]].isna().any().any():
        raise RuntimeError("Training dataset contains missing values")

    return df, feature_cols


def train_ranker(
    training_path: str,
    model_out: str,
    metrics_out: str,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train, evaluate, validate, and persist the XGBoost ranker."""
    df, feature_cols = load_training_data(training_path)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["overall_score"].to_numpy(dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=random_state
    )

    # XGBRanker expects non-negative integer relevance grades
    y_train_rank = np.clip(np.round(y_train), 0, 10).astype(np.int32)
    y_val_rank = np.clip(np.round(y_val), 0, 10).astype(np.int32)

    # Partition into pseudo query groups so pairwise ranking can learn
    group_size = 50
    n_train_groups = len(y_train_rank) // group_size
    n_val_groups = len(y_val_rank) // group_size
    n_train = n_train_groups * group_size
    n_val = n_val_groups * group_size

    X_train_g = X_train[:n_train]
    y_train_g = y_train_rank[:n_train]
    train_groups = [group_size] * n_train_groups

    X_val_g = X_val[:n_val]
    y_val_g = y_val_rank[:n_val]
    val_groups = [group_size] * n_val_groups

    print(f"\n[TRAIN] Samples: {len(X):,}  (train={n_train:,}, val={n_val:,})")
    print(f"  Query groups: train={n_train_groups}, val={n_val_groups} "
          f"(size={group_size})")
    print(f"  Features: {len(feature_cols)}")

    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        learning_rate=0.05,
        n_estimators=300,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    ranker.fit(
        X_train_g,
        y_train_g,
        group=train_groups,
        eval_set=[(X_val_g, y_val_g)],
        eval_group=[val_groups],
        verbose=False,
    )

    y_val_pred = ranker.predict(X_val)

    # Accuracy-style metrics against continuous teacher scores
    abs_err = np.abs(y_val - y_val_pred)
    tolerance_acc = float((abs_err <= 1.0).mean() * 100)
    tolerance_2_acc = float((abs_err <= 2.0).mean() * 100)
    exact_grade_acc = float(
        (np.round(y_val) == np.round(np.clip(y_val_pred, 0, 10))).mean() * 100
    )
    spearman = _spearman(y_val, y_val_pred)

    metrics = {
        "n_train": n_train,
        "n_val": n_val,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "target_column": "overall_score",
        "relevance_discretization": "round clip to [0, 10] int",
        "group_size": group_size,
        "n_train_groups": n_train_groups,
        "n_val_groups": n_val_groups,
        "val_rmse": round(float(np.sqrt(mean_squared_error(y_val, y_val_pred))), 4),
        "val_mae": round(float(mean_absolute_error(y_val, y_val_pred)), 4),
        "val_spearman": round(spearman, 4),
        "val_rank_correlation_pct": round(spearman * 100, 2),
        "val_accuracy_pct": round(tolerance_acc, 2),
        "val_accuracy_within_2pt_pct": round(tolerance_2_acc, 2),
        "val_exact_grade_accuracy_pct": round(exact_grade_acc, 2),
        "val_pred_min": round(float(y_val_pred.min()), 4),
        "val_pred_max": round(float(y_val_pred.max()), 4),
        "val_pred_mean": round(float(y_val_pred.mean()), 4),
        "best_iteration": int(ranker.best_iteration),
        "model_path": model_out,
    }

    model_path = Path(model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_model(str(model_path))

    # Post-training validation: reload and spot-check predictions
    loaded = xgb.XGBRanker()
    loaded.load_model(str(model_path))
    sample_pred = float(loaded.predict(X_val[:1])[0])
    reload_pred = float(ranker.predict(X_val[:1])[0])
    if not np.isclose(sample_pred, reload_pred, rtol=1e-5):
        raise RuntimeError("Model reload prediction mismatch")

    metrics["reload_check_passed"] = True
    metrics["sample_val_prediction"] = round(sample_pred, 4)
    metrics["sample_val_actual"] = round(float(y_val[0]), 4)

    metrics_path = Path(metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n=== Ranker Training Complete ===")
    print(f"  Validation RMSE:     {metrics['val_rmse']:.4f}")
    print(f"  Validation MAE:      {metrics['val_mae']:.4f}")
    print(f"  Validation Spearman: {metrics['val_spearman']:.4f}")
    print(f"  Model accuracy:      {metrics['val_accuracy_pct']:.1f}% "
          f"(within ±1.0 of teacher score)")
    print(f"  Rank correlation:    {metrics['val_rank_correlation_pct']:.1f}%")
    print(f"  Best iteration:      {metrics['best_iteration']}")
    print(f"  Pred range (val):    "
          f"{metrics['val_pred_min']:.2f} – {metrics['val_pred_max']:.2f}")
    print(f"  Reload check:        PASS")
    print(f"  Model saved:         {model_path}")
    print("=" * 40)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost Student Ranker")
    parser.add_argument(
        "--training",
        default="artifacts/training_dataset.parquet",
        help="Path to training dataset parquet",
    )
    parser.add_argument(
        "--model-out",
        default="artifacts/model.xgb",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--metrics-out",
        default="artifacts/training_metrics.json",
        help="Path to write evaluation metrics JSON",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation split fraction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    train_ranker(
        training_path=args.training,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        val_fraction=args.val_fraction,
        random_state=args.random_state,
    )
    print("\nTask 3.4 Complete. Model saved to artifacts/model.xgb")


if __name__ == "__main__":
    main()
