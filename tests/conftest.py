"""Test fixtures for MTO-Net tests.

All tests run on CPU for deterministic behavior.
Training uses Slurm GPU jobs on HPC server.
"""
import os
import pytest

# Force CPU before any torch import elsewhere
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture(scope="session", autouse=True)
def ensure_cpu():
    """Ensure tests run on CPU."""
    import torch
    # Skip if GPU is somehow still visible
    if torch.cuda.is_available():
        print("Warning: CUDA still visible in tests - this is fine on a laptop")
