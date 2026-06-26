"""
offline_pipeline/serialization/benchmark_artifacts.py

Benchmark script for measuring artifact load performance.
Tests serialization/deserialization speed and memory usage.

Outputs:
    artifacts/benchmark_results.json — Detailed performance metrics
"""

import os
import json
import pickle
import time
import pandas as pd
import psutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any


class PerformanceBenchmark:
    """Benchmark artifact loading performance."""
    
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = output_dir
        self.benchmark_path = os.path.join(output_dir, "benchmark_results.json")
        self.results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'benchmarks': {}
        }
    
    def _get_memory_usage(self):
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    def benchmark_parquet_load(self, parquet_path: str, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark parquet file loading."""
        print(f"\n[Benchmark] Testing parquet load performance ({iterations} iterations)...")
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
        
        # Warmup
        _ = pd.read_parquet(parquet_path)
        
        # Benchmark
        times = []
        for i in range(iterations):
            start = time.time()
            df = pd.read_parquet(parquet_path)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.4f}s ({len(df):,} rows)")
        
        results = {
            'file_path': parquet_path,
            'file_size_mb': round(file_size_mb, 3),
            'iterations': iterations,
            'times_seconds': [round(t, 4) for t in times],
            'min_time_seconds': round(min(times), 4),
            'max_time_seconds': round(max(times), 4),
            'mean_time_seconds': round(sum(times) / len(times), 4),
            'median_time_seconds': round(sorted(times)[len(times)//2], 4),
            'throughput_mb_per_second': round(file_size_mb / (sum(times) / len(times)), 2),
            'rows_loaded': len(df),
            'columns_loaded': len(df.columns)
        }
        
        print(f"  ✓ Mean load time: {results['mean_time_seconds']:.4f}s")
        print(f"  ✓ Throughput: {results['throughput_mb_per_second']:.2f} MB/s")
        
        return results
    
    def benchmark_pickle_load(self, pickle_path: str, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark pickle file loading."""
        print(f"\n[Benchmark] Testing pickle load performance ({iterations} iterations)...")
        
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        file_size_bytes = os.path.getsize(pickle_path)
        
        # Warmup
        with open(pickle_path, 'rb') as f:
            _ = pickle.load(f)
        
        # Benchmark
        times = []
        for i in range(iterations):
            start = time.time()
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed*1000:.3f}ms ({len(data)} items)")
        
        results = {
            'file_path': pickle_path,
            'file_size_bytes': file_size_bytes,
            'iterations': iterations,
            'times_seconds': [round(t, 6) for t in times],
            'min_time_seconds': round(min(times), 6),
            'max_time_seconds': round(max(times), 6),
            'mean_time_seconds': round(sum(times) / len(times), 6),
            'median_time_seconds': round(sorted(times)[len(times)//2], 6),
            'throughput_kb_per_second': round((file_size_bytes / 1024) / (sum(times) / len(times)), 2),
            'items_loaded': len(data)
        }
        
        print(f"  ✓ Mean load time: {results['mean_time_seconds']*1000:.3f}ms")
        print(f"  ✓ Throughput: {results['throughput_kb_per_second']:.2f} KB/s")
        
        return results
    
    def benchmark_metadata_load(self, metadata_path: str, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark metadata JSON loading."""
        print(f"\n[Benchmark] Testing metadata load performance ({iterations} iterations)...")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        file_size_bytes = os.path.getsize(metadata_path)
        
        # Benchmark
        times = []
        for i in range(iterations):
            start = time.time()
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed*1000:.3f}ms ({len(data.get('features', {}))} features)")
        
        results = {
            'file_path': metadata_path,
            'file_size_bytes': file_size_bytes,
            'iterations': iterations,
            'times_seconds': [round(t, 6) for t in times],
            'min_time_seconds': round(min(times), 6),
            'max_time_seconds': round(max(times), 6),
            'mean_time_seconds': round(sum(times) / len(times), 6),
            'median_time_seconds': round(sorted(times)[len(times)//2], 6),
            'features_in_metadata': len(data.get('features', {}))
        }
        
        print(f"  ✓ Mean load time: {results['mean_time_seconds']*1000:.3f}ms")
        
        return results
    
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("\n" + "="*70)
        print("ARTIFACT PERFORMANCE BENCHMARKS")
        print("="*70)
        
        try:
            # Benchmark parquet
            parquet_results = self.benchmark_parquet_load(
                os.path.join(self.output_dir, "features.parquet"),
                iterations=5
            )
            self.results['benchmarks']['features_parquet'] = parquet_results
        except Exception as e:
            print(f"✗ Parquet benchmark failed: {e}")
            self.results['benchmarks']['features_parquet'] = {'error': str(e)}
        
        try:
            # Benchmark pickle
            pickle_results = self.benchmark_pickle_load(
                os.path.join(self.output_dir, "honeypot_ids.pkl"),
                iterations=10
            )
            self.results['benchmarks']['honeypot_ids_pickle'] = pickle_results
        except Exception as e:
            print(f"✗ Pickle benchmark failed: {e}")
            self.results['benchmarks']['honeypot_ids_pickle'] = {'error': str(e)}
        
        try:
            # Benchmark metadata
            metadata_results = self.benchmark_metadata_load(
                os.path.join(self.output_dir, "feature_metadata.json"),
                iterations=10
            )
            self.results['benchmarks']['feature_metadata_json'] = metadata_results
        except Exception as e:
            print(f"✗ Metadata benchmark failed: {e}")
            self.results['benchmarks']['feature_metadata_json'] = {'error': str(e)}
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save benchmark results to JSON."""
        print(f"\n[Results] Saving benchmark results to {self.benchmark_path}...")
        with open(self.benchmark_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"  ✓ Results saved")
    
    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        if 'features_parquet' in self.results['benchmarks']:
            r = self.results['benchmarks']['features_parquet']
            if 'error' not in r:
                print(f"\nFeatures Parquet (features.parquet):")
                print(f"  File Size: {r['file_size_mb']:.2f} MB")
                print(f"  Rows: {r['rows_loaded']:,}")
                print(f"  Mean Load Time: {r['mean_time_seconds']:.4f}s")
                print(f"  Median Load Time: {r['median_time_seconds']:.4f}s")
                print(f"  Throughput: {r['throughput_mb_per_second']:.2f} MB/s")
                print(f"  Time Range: {r['min_time_seconds']:.4f}s - {r['max_time_seconds']:.4f}s")
        
        if 'honeypot_ids_pickle' in self.results['benchmarks']:
            r = self.results['benchmarks']['honeypot_ids_pickle']
            if 'error' not in r:
                print(f"\nHoneypot IDs Pickle (honeypot_ids.pkl):")
                print(f"  File Size: {r['file_size_bytes']} bytes")
                print(f"  Items: {r['items_loaded']:,}")
                print(f"  Mean Load Time: {r['mean_time_seconds']*1000:.3f}ms")
                print(f"  Median Load Time: {r['median_time_seconds']*1000:.3f}ms")
                print(f"  Throughput: {r['throughput_kb_per_second']:.2f} KB/s")
                print(f"  Time Range: {r['min_time_seconds']*1000:.3f}ms - {r['max_time_seconds']*1000:.3f}ms")
        
        if 'feature_metadata_json' in self.results['benchmarks']:
            r = self.results['benchmarks']['feature_metadata_json']
            if 'error' not in r:
                print(f"\nFeature Metadata (feature_metadata.json):")
                print(f"  File Size: {r['file_size_bytes']} bytes")
                print(f"  Features: {r['features_in_metadata']}")
                print(f"  Mean Load Time: {r['mean_time_seconds']*1000:.3f}ms")
                print(f"  Median Load Time: {r['median_time_seconds']*1000:.3f}ms")
                print(f"  Time Range: {r['min_time_seconds']*1000:.3f}ms - {r['max_time_seconds']*1000:.3f}ms")
        
        print("\n" + "="*70)


def main():
    try:
        benchmark = PerformanceBenchmark()
        results = benchmark.run_all_benchmarks()
        print("\n✓ All benchmarks completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
