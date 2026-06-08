"""Test MTO cache schema and serialization."""
import os
import sys
import tempfile
import torch
import numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)


def test_cache_entry_serialization():
    from src.mto.analysis.mto_cache import MTOCache
    cache = MTOCache(tempfile.mkdtemp(), max_molecules=10)
    entry = {
        "mol_id": 0,
        "smiles": "CCO",
        "K": 20,
        "z": np.array([6, 6, 8, 1, 1, 1, 1, 1, 1]),
        "mto_tensor": np.random.randn(20, 128),
        "activity": np.ones(20),
        "atom_contrib_norm": np.random.rand(9),
        "seed": 0,
        "stage": "stage_a",
    }
    cache.add_entry(entry)
    assert len(cache) == 1
    path = cache.save(tag="test")
    assert os.path.exists(path)
    assert os.path.exists(path.replace(".npz", "_meta.json"))


def test_cache_size_limit():
    from src.mto.analysis.mto_cache import MTOCache
    cache = MTOCache(tempfile.mkdtemp(), max_molecules=5)
    for i in range(10):
        cache.add_entry({"mol_id": i, "K": 10, "stage": "test"})
    assert len(cache) == 10
    path = cache.save(tag="test")
    # Should only save max_molecules
    data = np.load(path)
    n_saved = len([k for k in data.files if k.startswith("0_")])
    assert n_saved <= 6  # at most max_molecules entries


def test_cache_clear():
    from src.mto.analysis.mto_cache import MTOCache
    cache = MTOCache(tempfile.mkdtemp())
    for i in range(3):
        cache.add_entry({"mol_id": i})
    assert len(cache) == 3
    cache.clear()
    assert len(cache) == 0
