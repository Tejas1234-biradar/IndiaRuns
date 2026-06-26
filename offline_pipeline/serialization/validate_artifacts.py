"""
offline_pipeline/serialization/validate_artifacts.py

Comprehensive artifact validation script.
Verifies all serialized artifacts are intact and loadable.

Exit codes:
    0: All artifacts valid
    1: Validation failed
"""

import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def validate_all_artifacts(artifacts_dir: str = "artifacts") -> Dict[str, Any]:
    """Validate all serialization artifacts."""
    
    print("\n" + "="*70)
    print("ARTIFACT VALIDATION REPORT")
    print("="*70)
    
    results = {
        'valid': True,
        'artifacts': {}
    }
    
    # 1. Validate features.parquet
    print("\n[1/4] Validating features.parquet...")
    parquet_path = os.path.join(artifacts_dir, "features.parquet")
    try:
        df = pd.read_parquet(parquet_path)
        
        # Basic checks
        assert len(df) == 100000, f"Expected 100,000 rows, got {len(df)}"
        assert 'candidate_id' in df.columns, "candidate_id column missing"
        assert len(df['candidate_id'].unique()) == 100000, "Duplicate candidate IDs found"
        
        print(f"  ✓ Parquet file valid")
        print(f"    - Rows: {len(df):,}")
        print(f"    - Columns: {len(df.columns)}")
        print(f"    - File size: {os.path.getsize(parquet_path) / (1024*1024):.2f} MB")
        
        results['artifacts']['features_parquet'] = {
            'valid': True,
            'rows': len(df),
            'columns': len(df.columns),
            'file_size_bytes': os.path.getsize(parquet_path)
        }
        
    except Exception as e:
        print(f"  ✗ Parquet validation failed: {e}")
        results['artifacts']['features_parquet'] = {'valid': False, 'error': str(e)}
        results['valid'] = False
    
    # 2. Validate honeypot_ids.pkl
    print("\n[2/4] Validating honeypot_ids.pkl...")
    pickle_path = os.path.join(artifacts_dir, "honeypot_ids.pkl")
    try:
        with open(pickle_path, 'rb') as f:
            honeypots = pickle.load(f)
        
        assert isinstance(honeypots, set), f"Expected set, got {type(honeypots)}"
        assert len(honeypots) == 65, f"Expected 65 honeypots, got {len(honeypots)}"
        
        # Verify all are strings
        assert all(isinstance(h, str) for h in honeypots), "Non-string honeypot IDs found"
        
        print(f"  ✓ Pickle file valid")
        print(f"    - Type: {type(honeypots).__name__}")
        print(f"    - Count: {len(honeypots):,}")
        print(f"    - File size: {os.path.getsize(pickle_path)} bytes")
        
        results['artifacts']['honeypot_ids_pickle'] = {
            'valid': True,
            'count': len(honeypots),
            'file_size_bytes': os.path.getsize(pickle_path)
        }
        
    except Exception as e:
        print(f"  ✗ Pickle validation failed: {e}")
        results['artifacts']['honeypot_ids_pickle'] = {'valid': False, 'error': str(e)}
        results['valid'] = False
    
    # 3. Validate feature_metadata.json
    print("\n[3/4] Validating feature_metadata.json...")
    metadata_path = os.path.join(artifacts_dir, "feature_metadata.json")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert 'features' in metadata, "features key missing"
        assert len(metadata['features']) == 13, f"Expected 13 features, got {len(metadata['features'])}"
        assert 'total_rows' in metadata, "total_rows key missing"
        
        # Verify all features have required fields
        for fname, fdata in metadata['features'].items():
            assert 'dtype' in fdata, f"dtype missing for {fname}"
            assert 'null_count' in fdata, f"null_count missing for {fname}"
        
        print(f"  ✓ Metadata file valid")
        print(f"    - Total rows: {metadata['total_rows']:,}")
        print(f"    - Features documented: {len(metadata['features'])}")
        print(f"    - File size: {os.path.getsize(metadata_path)} bytes")
        
        results['artifacts']['feature_metadata_json'] = {
            'valid': True,
            'total_rows': metadata['total_rows'],
            'features': len(metadata['features']),
            'file_size_bytes': os.path.getsize(metadata_path)
        }
        
    except Exception as e:
        print(f"  ✗ Metadata validation failed: {e}")
        results['artifacts']['feature_metadata_json'] = {'valid': False, 'error': str(e)}
        results['valid'] = False
    
    # 4. Cross-artifact validation
    print("\n[4/4] Cross-artifact consistency checks...")
    try:
        assert len(df) == metadata['total_rows'], "Row count mismatch"
        
        # Check that honeypots are subset of candidates
        candidate_ids = set(df['candidate_id'])
        assert honeypots.issubset(candidate_ids), "Honeypots not in candidate set"
        
        valid_candidates = len(candidate_ids - honeypots)
        print(f"  ✓ Consistency checks passed")
        print(f"    - Total candidates: {len(candidate_ids):,}")
        print(f"    - Honeypots: {len(honeypots):,}")
        print(f"    - Valid candidates: {valid_candidates:,}")
        
        results['consistency'] = {
            'total_candidates': len(candidate_ids),
            'honeypot_count': len(honeypots),
            'valid_candidates': valid_candidates
        }
        
    except Exception as e:
        print(f"  ✗ Consistency validation failed: {e}")
        results['consistency'] = {'valid': False, 'error': str(e)}
        results['valid'] = False
    
    # Summary
    print("\n" + "="*70)
    if results['valid']:
        print("✓ ALL ARTIFACTS VALID")
    else:
        print("✗ VALIDATION FAILED")
    print("="*70)
    
    return results


def main():
    try:
        results = validate_all_artifacts()
        
        # Print detailed summary
        print("\nDetailed Results:")
        print("-" * 70)
        
        all_valid = True
        for artifact_name, artifact_result in results['artifacts'].items():
            status = "✓" if artifact_result.get('valid', False) else "✗"
            print(f"{status} {artifact_name}: {'VALID' if artifact_result.get('valid') else 'INVALID'}")
            
            if not artifact_result.get('valid'):
                all_valid = False
                if 'error' in artifact_result:
                    print(f"   Error: {artifact_result['error']}")
        
        if 'consistency' in results:
            consistency = results['consistency']
            if 'error' not in consistency:
                print(f"✓ Consistency checks passed")
                print(f"   Total candidates: {consistency['total_candidates']:,}")
                print(f"   Honeypots: {consistency['honeypot_count']:,}")
                print(f"   Valid: {consistency['valid_candidates']:,}")
            else:
                all_valid = False
        
        print("-" * 70)
        
        if all_valid:
            print("\n✓ All validation checks passed successfully!")
            return 0
        else:
            print("\n✗ Validation failed - check errors above")
            return 1
            
    except Exception as e:
        print(f"\n✗ Validation script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
