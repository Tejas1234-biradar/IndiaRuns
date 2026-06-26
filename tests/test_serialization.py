"""
tests/test_serialization.py

Test suite for feature serialization pipeline.
Tests round-trip serialization, integrity checks, and performance.
"""

import os
import json
import pickle
import tempfile
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from offline_pipeline.serialization.artifact_serializer import (
    FeatureMatrixSerializer,
    HoneypotSerializer,
    MetadataGenerator,
    ArtifactValidator,
    SerializationPipeline
)


class TestFeatureMatrixSerializer:
    """Test feature matrix serialization."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample feature dataframe."""
        return pd.DataFrame({
            'candidate_id': [f'cand_{i}' for i in range(100)],
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randint(0, 100, 100),
            'feature_4': np.random.uniform(0, 1, 100)
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_serialize_features(self, sample_dataframe, temp_dir):
        """Test basic feature serialization."""
        serializer = FeatureMatrixSerializer(temp_dir)
        path, metadata = serializer.serialize_features(sample_dataframe)
        
        assert os.path.exists(path)
        assert metadata['num_candidates'] == 100
        assert metadata['total_columns'] == 5
        assert 'file_size_mb' in metadata
        assert 'serialization_time_seconds' in metadata
    
    def test_serialize_empty_dataframe(self, temp_dir):
        """Test serialization of empty dataframe."""
        serializer = FeatureMatrixSerializer(temp_dir)
        empty_df = pd.DataFrame({'candidate_id': []})
        
        with pytest.raises(ValueError):
            serializer.serialize_features(empty_df)
    
    def test_serialize_missing_candidate_id(self, temp_dir):
        """Test serialization without candidate_id column."""
        serializer = FeatureMatrixSerializer(temp_dir)
        df = pd.DataFrame({'feature_1': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            serializer.serialize_features(df)
    
    def test_round_trip_serialization(self, sample_dataframe, temp_dir):
        """Test serialization and deserialization round-trip."""
        serializer = FeatureMatrixSerializer(temp_dir)
        path, _ = serializer.serialize_features(sample_dataframe)
        
        # Load back
        loaded_df = pd.read_parquet(path)
        
        # Verify integrity
        assert len(loaded_df) == len(sample_dataframe)
        assert list(loaded_df.columns) == list(sample_dataframe.columns)
        assert set(loaded_df['candidate_id']) == set(sample_dataframe['candidate_id'])


class TestHoneypotSerializer:
    """Test honeypot ID serialization."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_serialize_honeypots(self, temp_dir):
        """Test honeypot serialization."""
        honeypots = {'honeypot_1', 'honeypot_2', 'honeypot_3'}
        serializer = HoneypotSerializer(temp_dir)
        
        path, metadata = serializer.serialize_honeypots(honeypots)
        
        assert os.path.exists(path)
        assert metadata['num_honeypots'] == 3
        assert 'file_size_bytes' in metadata
        assert 'serialization_time_seconds' in metadata
    
    def test_serialize_empty_honeypots(self, temp_dir):
        """Test serialization of empty honeypot set."""
        serializer = HoneypotSerializer(temp_dir)
        
        with pytest.raises(ValueError):
            serializer.serialize_honeypots(set())
    
    def test_round_trip_honeypots(self, temp_dir):
        """Test honeypot serialization round-trip."""
        original_honeypots = {'hp_1', 'hp_2', 'hp_3', 'hp_4', 'hp_5'}
        serializer = HoneypotSerializer(temp_dir)
        
        path, _ = serializer.serialize_honeypots(original_honeypots)
        
        # Load back
        with open(path, 'rb') as f:
            loaded_honeypots = pickle.load(f)
        
        assert loaded_honeypots == original_honeypots


class TestMetadataGenerator:
    """Test metadata generation."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe."""
        return pd.DataFrame({
            'candidate_id': ['c1', 'c2', 'c3'],
            'feature_1': [1.0, 2.0, np.nan],
            'feature_2': [10, 20, 30],
            'feature_3': ['a', 'b', 'c']
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_generate_metadata(self, sample_dataframe, temp_dir):
        """Test metadata generation."""
        generator = MetadataGenerator(temp_dir)
        path, metadata = generator.generate_metadata(sample_dataframe)
        
        assert os.path.exists(path)
        assert metadata['total_columns'] == 4
        assert metadata['total_rows'] == 3
        assert len(metadata['features']) == 4
    
    def test_metadata_includes_statistics(self, sample_dataframe, temp_dir):
        """Test that metadata includes statistics for numeric columns."""
        generator = MetadataGenerator(temp_dir)
        _, metadata = generator.generate_metadata(sample_dataframe)
        
        feature_1_meta = metadata['features']['feature_1']
        assert 'min' in feature_1_meta
        assert 'max' in feature_1_meta
        assert 'mean' in feature_1_meta
        assert 'null_count' in feature_1_meta
        assert feature_1_meta['null_count'] == 1
    
    def test_metadata_json_format(self, sample_dataframe, temp_dir):
        """Test that metadata is valid JSON."""
        generator = MetadataGenerator(temp_dir)
        path, _ = generator.generate_metadata(sample_dataframe)
        
        # Load JSON
        with open(path, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert 'features' in data
        assert 'total_rows' in data


class TestArtifactValidator:
    """Test artifact validation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_validate_valid_parquet(self, temp_dir):
        """Test validation of valid parquet file."""
        # Create test parquet
        df = pd.DataFrame({
            'candidate_id': ['c1', 'c2', 'c3'],
            'feature_1': [1.0, 2.0, 3.0]
        })
        parquet_path = os.path.join(temp_dir, 'test.parquet')
        df.to_parquet(parquet_path)
        
        # Validate
        validator = ArtifactValidator()
        results = validator.validate_parquet(parquet_path)
        
        assert results['valid']
        assert results['num_rows'] == 3
        assert results['num_columns'] == 2
        assert results['has_candidate_id']
        assert 'load_time_seconds' in results
    
    def test_validate_missing_parquet(self):
        """Test validation of missing parquet file."""
        validator = ArtifactValidator()
        results = validator.validate_parquet('/nonexistent/path.parquet')
        
        assert not results['valid']
        assert 'error' in results
    
    def test_validate_valid_pickle(self, temp_dir):
        """Test validation of valid pickle file."""
        # Create test pickle
        data = {'item1', 'item2', 'item3'}
        pickle_path = os.path.join(temp_dir, 'test.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Validate
        validator = ArtifactValidator()
        results = validator.validate_pickle(pickle_path)
        
        assert results['valid']
        assert results['data_type'] == 'set'
        assert results['size'] == 3
        assert 'load_time_seconds' in results
    
    def test_validate_missing_pickle(self):
        """Test validation of missing pickle file."""
        validator = ArtifactValidator()
        results = validator.validate_pickle('/nonexistent/path.pkl')
        
        assert not results['valid']
        assert 'error' in results


class TestSerializationPipeline:
    """Test complete serialization pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample feature matrix and honeypot IDs."""
        df = pd.DataFrame({
            'candidate_id': [f'c_{i}' for i in range(100)],
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        })
        honeypots = {'c_0', 'c_1', 'c_2'}
        return df, honeypots
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_pipeline_complete_run(self, sample_data, temp_dir):
        """Test complete pipeline execution."""
        df, honeypots = sample_data
        pipeline = SerializationPipeline(temp_dir)
        
        results = pipeline.run(df, honeypots)
        
        assert 'features' in results
        assert 'honeypots' in results
        assert 'metadata' in results
        assert 'validation' in results
        assert 'consistency' in results
        
        # Verify artifacts were created
        assert os.path.exists(os.path.join(temp_dir, 'features.parquet'))
        assert os.path.exists(os.path.join(temp_dir, 'honeypot_ids.pkl'))
        assert os.path.exists(os.path.join(temp_dir, 'feature_metadata.json'))
    
    def test_pipeline_consistency_checks(self, sample_data, temp_dir):
        """Test consistency checks in pipeline."""
        df, honeypots = sample_data
        pipeline = SerializationPipeline(temp_dir)
        
        results = pipeline.run(df, honeypots)
        
        consistency = results['consistency']
        assert consistency['total_candidates'] == 100
        assert consistency['honeypot_count'] == 3
        assert consistency['valid_candidates'] == 97


class TestSerializationIntegration:
    """Integration tests for serialization."""
    
    def test_full_serialization_workflow(self):
        """Test complete serialization workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            df = pd.DataFrame({
                'candidate_id': [f'c_{i}' for i in range(50)],
                'years_exp': np.random.uniform(0, 10, 50),
                'skills_count': np.random.randint(1, 20, 50),
                'score': np.random.uniform(0, 100, 50)
            })
            honeypots = {'c_0', 'c_1', 'c_2'}
            
            # Run pipeline
            pipeline = SerializationPipeline(tmpdir)
            results = pipeline.run(df, honeypots)
            
            # Verify results
            assert results['features']['metadata']['num_candidates'] == 50
            assert results['honeypots']['metadata']['num_honeypots'] == 3
            assert results['consistency']['valid_candidates'] == 47
            
            # Verify validation passed
            assert results['validation']['features_parquet']['valid']
            assert results['validation']['honeypots_pickle']['valid']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
