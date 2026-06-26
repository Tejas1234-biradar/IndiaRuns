"""
offline_pipeline/serialization/run_serialization.py

Main execution script for Feature Serialization Pipeline.
Loads feature matrix from parquet, honeypot IDs from pickle,
and runs complete serialization workflow.

Usage:
    python offline_pipeline/serialization/run_serialization.py
"""

import os
import sys
import json
import pickle
import pandas as pd
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from serialization.artifact_serializer import SerializationPipeline
from feature_engineering.feature_schema import FEATURE_SCHEMA


# --- PATHS ---
ARTIFACTS_DIR = "artifacts"
FEATURE_INPUT_PATH = os.path.join(ARTIFACTS_DIR, "candidate_features.parquet")
HONEYPOT_INPUT_PATH = os.path.join(ARTIFACTS_DIR, "honeypot_ids.pkl")


def load_feature_matrix() -> pd.DataFrame:
    """Load pre-computed feature matrix."""
    print(f"\n[Input] Loading feature matrix from {FEATURE_INPUT_PATH}...")
    
    if not os.path.exists(FEATURE_INPUT_PATH):
        raise FileNotFoundError(f"Feature matrix not found: {FEATURE_INPUT_PATH}")
        
    df = pd.read_parquet(FEATURE_INPUT_PATH)
    print(f"  ✓ Loaded {len(df):,} candidates with {len(df.columns)} columns")
    
    return df


def load_honeypot_ids() -> set:
    """Load honeypot candidate IDs."""
    print(f"\n[Input] Loading honeypot IDs from {HONEYPOT_INPUT_PATH}...")
    
    if not os.path.exists(HONEYPOT_INPUT_PATH):
        raise FileNotFoundError(f"Honeypot IDs not found: {HONEYPOT_INPUT_PATH}")
        
    with open(HONEYPOT_INPUT_PATH, 'rb') as f:
        honeypot_ids = pickle.load(f)
        
    if isinstance(honeypot_ids, dict):
        # Handle both set and dict formats
        honeypot_ids = set(honeypot_ids.keys()) if honeypot_ids else set()
    elif not isinstance(honeypot_ids, set):
        honeypot_ids = set(honeypot_ids) if honeypot_ids else set()
        
    print(f"  ✓ Loaded {len(honeypot_ids):,} honeypot IDs")
    
    return honeypot_ids


def main():
    """Main execution."""
    try:
        # Load inputs
        feature_df = load_feature_matrix()
        honeypot_ids = load_honeypot_ids()
        
        # Run pipeline
        pipeline = SerializationPipeline(output_dir=ARTIFACTS_DIR)
        results = pipeline.run(
            feature_df=feature_df,
            honeypot_ids=honeypot_ids,
            feature_schema=FEATURE_SCHEMA
        )
        
        # Save results summary
        results_path = os.path.join(ARTIFACTS_DIR, "serialization_results.json")
        print(f"\n[Summary] Writing results to {results_path}...")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Results saved")
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        if 'features' in results and 'metadata' in results['features']:
            meta = results['features']['metadata']
            print(f"Features Parquet:")
            print(f"  - Candidates: {meta['num_candidates']:,}")
            print(f"  - Columns: {meta['total_columns']}")
            print(f"  - File size: {meta['file_size_mb']:.2f} MB")
            
        if 'honeypots' in results and 'metadata' in results['honeypots']:
            meta = results['honeypots']['metadata']
            print(f"\nHoneypot IDs Pickle:")
            print(f"  - Count: {meta['num_honeypots']:,}")
            
        if 'validation' in results:
            if 'features_parquet' in results['validation']:
                val = results['validation']['features_parquet']
                if val.get('valid'):
                    print(f"\nFeatures Parquet Validation:")
                    print(f"  - ✓ Load time: {val['load_time_seconds']:.3f}s")
                    print(f"  - ✓ Rows: {val['num_rows']:,}")
                    print(f"  - ✓ Columns: {val['num_columns']}")
                    
            if 'honeypots_pickle' in results['validation']:
                val = results['validation']['honeypots_pickle']
                if val.get('valid'):
                    print(f"\nHoneypots Pickle Validation:")
                    print(f"  - ✓ Load time: {val['load_time_seconds']:.3f}s")
                    print(f"  - ✓ Type: {val['data_type']}")
                    print(f"  - ✓ Size: {val['size']:,}")
        
        print("\n✓ Serialization pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
