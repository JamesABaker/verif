#!/usr/bin/env python3
"""
Download and cache the HC3 dataset locally for evaluation.
This avoids issues with the dataset loading scripts and provides offline access.
"""
import os
import sys
from pathlib import Path

from datasets import load_dataset

# Dataset cache directory (can be customized via HC3_CACHE_DIR env var)
CACHE_DIR = Path(os.getenv("HC3_CACHE_DIR", "/data/hc3_dataset"))


def download_hc3():
    """Download HC3 dataset and save locally."""
    print(f"Downloading HC3 dataset to {CACHE_DIR}...")

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Load from parquet revision (most reliable)
        print("Loading from parquet revision...")
        dataset = load_dataset(
            "Hello-SimpleAI/HC3",
            "default",
            split="train",
            revision="refs/convert/parquet",
            cache_dir=str(CACHE_DIR),
        )

        print("✓ Successfully downloaded HC3 dataset")
        print("  Config: default")
        print("  Split: train")
        print(f"  Rows: {len(dataset):,}")
        print(f"  Cached at: {CACHE_DIR}")

        # Verify the structure
        print(f"\n  Sample keys: {list(dataset[0].keys())}")

        return True

    except Exception as e:
        print(f"✗ Failed to download dataset: {e}")
        return False


if __name__ == "__main__":
    success = download_hc3()
    sys.exit(0 if success else 1)
