#!/usr/bin/env python3
"""
verify_pipeline.py
Cross-platform validation workflow script.
Builds the docker image, executes candidate ranking, and validates the output.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add tests directory to path to import validate_submission
sys.path.insert(0, str(Path(__file__).parent / "tests"))
try:
    from validate_submission import validate_submission
except ImportError:
    print("Error: Could not import validate_submission.py from tests/ directory.", file=sys.stderr)
    sys.exit(1)


def run_command(cmd, shell=False):
    """Run a system command and stream output, returning exit code."""
    print(f"\nRunning command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        print(result.stdout)
        return 0
    except subprocess.CalledProcessError as e:
        print("Command failed with output:")
        print(e.stdout)
        return e.returncode


def run_local_pipeline(candidates, out):
    print("\n=== Running Local Validation Pipeline ===")
    
    # Run rank.py locally
    cmd = [sys.executable, "rank.py", "--candidates", candidates, "--out", out]
    exit_code = run_command(cmd)
    if exit_code != 0:
        print("Local rank.py execution failed.", file=sys.stderr)
        sys.exit(exit_code)
        
    validate_output_file(out)


def run_docker_pipeline(candidates, out):
    print("\n=== Running Docker Validation Pipeline ===")
    
    # 1. Build Docker image
    image_tag = "indiaruns-runtime"
    build_cmd = ["docker", "build", "-t", image_tag, "-f", "Dockerfile", "."]
    exit_code = run_command(build_cmd)
    if exit_code != 0:
        print("Docker build failed.", file=sys.stderr)
        sys.exit(exit_code)
        
    # 2. Setup mount paths for candidates and output CSV
    abs_candidates = Path(candidates).resolve()
    abs_out = Path(out).resolve()
    
    # The parent directories must be mounted
    candidates_dir = abs_candidates.parent
    out_dir = abs_out.parent
    
    # Target paths inside container
    container_candidates = f"/mount/candidates/{abs_candidates.name}"
    container_out = f"/mount/out/{abs_out.name}"
    
    # Run container
    # Mount candidates folder as read-only, mount out folder as read-write
    run_cmd = [
        "docker", "run", "--rm",
        "-v", f"{candidates_dir}:/mount/candidates:ro",
        "-v", f"{out_dir}:/mount/out:rw",
        image_tag,
        "--candidates", container_candidates,
        "--out", container_out
    ]
    
    exit_code = run_command(run_cmd)
    if exit_code != 0:
        print("Docker rank.py execution failed.", file=sys.stderr)
        sys.exit(exit_code)
        
    validate_output_file(out)


def validate_output_file(out_csv_path):
    print(f"\n=== Running validate_submission.py on {out_csv_path} ===")
    errors = validate_submission(out_csv_path)
    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("Validation PASSED successfully! The output CSV is compliant.")


def main():
    parser = argparse.ArgumentParser(description="Verify candidate discovery pipeline & submission validation")
    parser.add_argument("--docker", action="store_true", help="Run validation inside a built Docker container")
    parser.add_argument("--candidates", default="data/candidates.jsonl", help="Path to input candidates JSONL file")
    parser.add_argument("--out", default="submission.csv", help="Path to write validation submission CSV")
    
    args = parser.parse_args()
    
    # Ensure candidates file exists
    if not Path(args.candidates).exists():
        print(f"Error: Candidates file not found at '{args.candidates}'", file=sys.stderr)
        sys.exit(1)
        
    if args.docker:
        run_docker_pipeline(args.candidates, args.out)
    else:
        run_local_pipeline(args.candidates, args.out)


if __name__ == "__main__":
    main()
