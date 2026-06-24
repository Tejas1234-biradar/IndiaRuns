import os
import json
import xgboost as xgb
import pandas as pd
import numpy as np
from generate_labels import load_candidates, compute_mock_evaluation
from build_features import build_feature_matrix

def main():
    print("Starting model training pipeline...")
    
    # 1. Load candidates
    input_file = "candidates.jsonl"
    print(f"Loading candidate records from {input_file}...")
    candidates = load_candidates(input_file)
    
    # Take a 5,000 candidate slice for training (large enough for robust training)
    train_slice = candidates[:5000]
    print(f"Loaded {len(train_slice)} candidates for model training.")
    
    # 2. Grade candidates using LLM mock teacher rubric
    print("Generating training labels (mock teacher scores)...")
    grades = []
    for c in train_slice:
        cid = c.get("candidate_id")
        grade = compute_mock_evaluation(c)
        grades.append({
            "candidate_id": cid,
            "overall_score": grade.get("overall_score"),
            "hard_reject": float(grade.get("hard_reject"))
        })
        
    df_labels = pd.DataFrame(grades)
    
    # 3. Build features matrix
    print("Extracting candidate feature matrix...")
    df_features = build_feature_matrix(train_slice)
    
    # Merge labels and features
    df_data = pd.merge(df_features, df_labels, on="candidate_id")
    
    # Define features and target
    # Exclude IDs, target overall_score, and hard_reject helper label
    feature_cols = [col for col in df_features.columns if col != "candidate_id"]
    
    X = df_data[feature_cols]
    y = df_data["overall_score"]
    
    # Save the feature sequence explicitly to models/feature_order.json and baseline means to models/feature_means.json
    os.makedirs("models", exist_ok=True)
    with open("models/feature_order.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    
    feature_means = X.mean().to_dict()
    with open("models/feature_means.json", "w") as f:
        json.dump(feature_means, f, indent=2)
    print(f"Saved feature order sequence and baseline means for {len(feature_cols)} features.")
    
    # 4. Train the XGBoost Regressor model
    print("Training XGBoost Regressor model...")
    # Using conservative default parameters for generalizability
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # 5. Evaluate and report training statistics
    preds = model.predict(X)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    print(f"Model Training RMSE: {rmse:.4f}")
    
    # Feature Importances
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Feature Importances:")
    for feat, imp in feat_imp[:10]:
        print(f"  {feat}: {imp:.4f}")
        
    # 6. Save serialized weights
    model_path = "models/model.xgb"
    model.save_model(model_path)
    print(f"Saved trained XGBoost model to {model_path}")
    
    # Save feature importance csv for SHAP approximations
    df_imp = pd.DataFrame(feat_imp, columns=["feature", "importance"])
    df_imp.to_csv("reports/feature_importance.csv", index=False)
    print("Saved feature importances to reports/feature_importance.csv")

if __name__ == "__main__":
    main()
