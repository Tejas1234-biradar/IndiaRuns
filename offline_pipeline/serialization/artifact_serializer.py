"""
offline_pipeline/serialization/artifact_serializer.py

Task 4.3 — Feature Serialization Pipeline
Serializes pre-computed candidate features and honeypot IDs into compact,
loadable formats for the ranking system runtime.

Outputs:
    artifacts/features.parquet      — Compressed feature matrix (candidate_id + features)
    artifacts/honeypot_ids.pkl      — Pickled set of honeypot candidate IDs
    artifacts/feature_metadata.json — Schema metadata (columns, types, descriptions)
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Set, Any, Tuple
import time


class FeatureMatrixSerializer:
    """Handles serialization of feature matrices to parquet format."""
    
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.feature_parquet_path = os.path.join(output_dir, "features.parquet")
        
    def serialize_features(self, df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """
        Serialize feature matrix to parquet format.
        
        Args:
            df: DataFrame with candidate_id and feature columns
            
        Returns:
            Tuple of (output_path, metadata_dict)
        """
        print(f"\n[Serialization] Serializing feature matrix...")
        
        # Validate input
        if df.empty:
            raise ValueError("Feature DataFrame is empty")
        if 'candidate_id' not in df.columns:
            raise ValueError("candidate_id column required")
            
        # Ensure memory efficiency - downcast floats
        float_cols = df.select_dtypes(include=['float64']).columns
        if len(float_cols) > 0:
            print(f"  Downcasting {len(float_cols)} float64 columns to float32...")
            df = df.copy()
            df[float_cols] = df[float_cols].astype('float32')
        
        # Serialize to parquet
        print(f"  Writing to {self.feature_parquet_path}...")
        start_time = time.time()
        df.to_parquet(
            self.feature_parquet_path,
            engine='pyarrow',
            index=False,
            compression='snappy'  # Balanced compression
        )
        elapsed = time.time() - start_time
        
        # Collect metadata
        file_size_bytes = os.path.getsize(self.feature_parquet_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        metadata = {
            'num_candidates': len(df),
            'num_features': len(df.columns) - 1,  # Exclude candidate_id
            'total_columns': len(df.columns),
            'file_size_bytes': file_size_bytes,
            'file_size_mb': round(file_size_mb, 2),
            'serialization_time_seconds': round(elapsed, 3),
            'compression': 'snappy',
            'format': 'parquet'
        }
        
        print(f"  ✓ Serialized {len(df):,} candidates with {len(df.columns)} columns")
        print(f"  ✓ File size: {file_size_mb:.2f} MB (serialized in {elapsed:.3f}s)")
        
        return self.feature_parquet_path, metadata


class HoneypotSerializer:
    """Handles serialization of honeypot candidate IDs."""
    
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.honeypot_pkl_path = os.path.join(output_dir, "honeypot_ids.pkl")
        
    def serialize_honeypots(self, honeypot_ids: Set[str]) -> Tuple[str, Dict[str, Any]]:
        """
        Serialize honeypot IDs to pickle format.
        
        Args:
            honeypot_ids: Set of honeypot candidate IDs
            
        Returns:
            Tuple of (output_path, metadata_dict)
        """
        print(f"\n[Serialization] Serializing honeypot IDs...")
        
        if not honeypot_ids:
            raise ValueError("No honeypot IDs provided")
            
        print(f"  Writing {len(honeypot_ids):,} honeypot IDs to {self.honeypot_pkl_path}...")
        start_time = time.time()
        
        with open(self.honeypot_pkl_path, 'wb') as f:
            pickle.dump(honeypot_ids, f)
            
        elapsed = time.time() - start_time
        file_size_bytes = os.path.getsize(self.honeypot_pkl_path)
        
        metadata = {
            'num_honeypots': len(honeypot_ids),
            'file_size_bytes': file_size_bytes,
            'serialization_time_seconds': round(elapsed, 3),
            'format': 'pickle'
        }
        
        print(f"  ✓ Serialized {len(honeypot_ids):,} honeypot IDs")
        print(f"  ✓ File size: {file_size_bytes} bytes")
        
        return self.honeypot_pkl_path, metadata


class MetadataGenerator:
    """Generates feature schema metadata."""
    
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metadata_path = os.path.join(output_dir, "feature_metadata.json")
        
    def generate_metadata(self, df: pd.DataFrame, feature_schema: Dict[str, Dict] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate feature schema metadata from DataFrame.
        
        Args:
            df: DataFrame to analyze
            feature_schema: Optional schema dict with descriptions
            
        Returns:
            Tuple of (metadata_path, metadata_dict)
        """
        print(f"\n[Metadata] Generating feature schema metadata...")
        
        metadata = {
            'generated_at': datetime.utcnow().isoformat(),
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'features': {}
        }
        
        # Document each column
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isna().sum()
            
            feature_info = {
                'dtype': dtype,
                'pandas_dtype': str(df[col].dtype),
                'null_count': int(null_count),
                'null_percentage': round(100 * null_count / len(df), 2),
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_info['min'] = float(df[col].min()) if not df[col].isna().all() else None
                feature_info['max'] = float(df[col].max()) if not df[col].isna().all() else None
                feature_info['mean'] = float(df[col].mean()) if not df[col].isna().all() else None
                
            # Add description from schema if available
            if feature_schema and col in feature_schema:
                feature_info['description'] = feature_schema[col].get('description', '')
                feature_info['imputation_rule'] = feature_schema[col].get('imputation', 'unknown')
                
            metadata['features'][col] = feature_info
        
        # Write metadata
        print(f"  Writing metadata to {self.metadata_path}...")
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"  ✓ Metadata written with {len(metadata['features'])} features")
        
        return self.metadata_path, metadata


class ArtifactValidator:
    """Validates serialized artifacts for integrity."""
    
    @staticmethod
    def validate_parquet(parquet_path: str) -> Dict[str, Any]:
        """
        Validate parquet file integrity.
        
        Args:
            parquet_path: Path to parquet file
            
        Returns:
            Validation results dictionary
        """
        print(f"\n[Validation] Validating parquet file...")
        
        try:
            start_time = time.time()
            df = pd.read_parquet(parquet_path)
            load_time = time.time() - start_time
            
            results = {
                'valid': True,
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'load_time_seconds': round(load_time, 3),
                'file_size_mb': os.path.getsize(parquet_path) / (1024 * 1024),
                'has_candidate_id': 'candidate_id' in df.columns,
                'candidate_ids_unique': df['candidate_id'].nunique() if 'candidate_id' in df.columns else 0,
                'null_values_per_column': df.isna().sum().to_dict()
            }
            
            print(f"  ✓ Parquet valid: {len(df):,} rows, {len(df.columns)} columns")
            print(f"  ✓ Load time: {load_time:.3f}s")
            
            return results
            
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            return {
                'valid': False,
                'error': str(e)
            }
    
    @staticmethod
    def validate_pickle(pickle_path: str) -> Dict[str, Any]:
        """
        Validate pickle file integrity.
        
        Args:
            pickle_path: Path to pickle file
            
        Returns:
            Validation results dictionary
        """
        print(f"\n[Validation] Validating pickle file...")
        
        try:
            start_time = time.time()
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            load_time = time.time() - start_time
            
            results = {
                'valid': True,
                'data_type': type(data).__name__,
                'size': len(data) if hasattr(data, '__len__') else 0,
                'load_time_seconds': round(load_time, 3),
                'file_size_bytes': os.path.getsize(pickle_path)
            }
            
            print(f"  ✓ Pickle valid: {type(data).__name__} with {len(data) if hasattr(data, '__len__') else '?'} items")
            print(f"  ✓ Load time: {load_time:.3f}s")
            
            return results
            
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            return {
                'valid': False,
                'error': str(e)
            }


class SerializationPipeline:
    """Orchestrates the complete serialization workflow."""
    
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = output_dir
        self.feature_serializer = FeatureMatrixSerializer(output_dir)
        self.honeypot_serializer = HoneypotSerializer(output_dir)
        self.metadata_gen = MetadataGenerator(output_dir)
        self.validator = ArtifactValidator()
        
    def run(self, feature_df: pd.DataFrame, honeypot_ids: Set[str], feature_schema: Dict = None) -> Dict[str, Any]:
        """
        Run complete serialization pipeline.
        
        Args:
            feature_df: Feature matrix DataFrame
            honeypot_ids: Set of honeypot candidate IDs
            feature_schema: Optional feature schema for metadata
            
        Returns:
            Complete results dictionary
        """
        print("\n" + "="*70)
        print("FEATURE SERIALIZATION PIPELINE")
        print("="*70)
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'features': {},
            'honeypots': {},
            'metadata': {},
            'validation': {}
        }
        
        # Phase 1: Serialize features
        try:
            feature_path, feature_meta = self.feature_serializer.serialize_features(feature_df)
            results['features'] = {
                'path': feature_path,
                'metadata': feature_meta
            }
        except Exception as e:
            print(f"✗ Feature serialization failed: {e}")
            results['features']['error'] = str(e)
            
        # Phase 2: Serialize honeypots
        try:
            honeypot_path, honeypot_meta = self.honeypot_serializer.serialize_honeypots(honeypot_ids)
            results['honeypots'] = {
                'path': honeypot_path,
                'metadata': honeypot_meta
            }
        except Exception as e:
            print(f"✗ Honeypot serialization failed: {e}")
            results['honeypots']['error'] = str(e)
            
        # Phase 3: Generate metadata
        try:
            metadata_path, metadata = self.metadata_gen.generate_metadata(feature_df, feature_schema)
            results['metadata'] = {
                'path': metadata_path,
                'summary': metadata
            }
        except Exception as e:
            print(f"✗ Metadata generation failed: {e}")
            results['metadata']['error'] = str(e)
            
        # Phase 4: Validate all artifacts
        try:
            if 'path' in results['features']:
                results['validation']['features_parquet'] = self.validator.validate_parquet(
                    results['features']['path']
                )
            if 'path' in results['honeypots']:
                results['validation']['honeypots_pickle'] = self.validator.validate_pickle(
                    results['honeypots']['path']
                )
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            results['validation']['error'] = str(e)
            
        # Phase 5: Verify candidate ID consistency
        print(f"\n[Consistency] Verifying candidate ID consistency...")
        try:
            parquet_ids = set(feature_df['candidate_id'].unique())
            print(f"  ✓ Candidates in feature matrix: {len(parquet_ids):,}")
            print(f"  ✓ Honeypots detected: {len(honeypot_ids):,}")
            
            valid_candidates = parquet_ids - honeypot_ids
            print(f"  ✓ Valid candidates (feature matrix - honeypots): {len(valid_candidates):,}")
            
            results['consistency'] = {
                'total_candidates': len(parquet_ids),
                'honeypot_count': len(honeypot_ids),
                'valid_candidates': len(valid_candidates)
            }
        except Exception as e:
            print(f"✗ Consistency check failed: {e}")
            results['consistency']['error'] = str(e)
            
        print("\n" + "="*70)
        print("SERIALIZATION COMPLETE")
        print("="*70)
        
        return results
