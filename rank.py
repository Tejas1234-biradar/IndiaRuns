#!/usr/bin/env python3
"""
Wrapper entrypoint for candidate ranking pipeline.
Delegates execution to runtime_pipeline/rank.py.
"""
import sys
from runtime_pipeline.rank import main

if __name__ == "__main__":
    main()
