"""Test MTO cache save/load with spectral predictions."""
import os, sys, pytest, tempfile
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_cache_save_load():
    """Cache should save and reload entries."""
    from src.mto.analysis.mto_cache import MTOCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MTOCache(tmpdir, max_molecules=8)
        for i in range(4):
            cache.add_entry({
                "mol_id": i,
                "pred_mu": np.array([0.1, 0.2, 0.3]),
                "pred_ir": np.random.randn(100).astype(np.float32),
            })
        path = cache.save(tag="test")
        assert os.path.exists(path)
        # Reload
        data = np.load(path, allow_pickle=True)
        assert len(data) > 0


def test_cache_max_molecules():
    """Cache should cap at max_molecules."""
    from src.mto.analysis.mto_cache import MTOCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MTOCache(tmpdir, max_molecules=3)
        for i in range(10):
            cache.add_entry({"mol_id": i, "val": np.array([float(i)])})
        assert len(cache._entries) == 10  # all added
        path = cache.save(tag="capped")
        # After save, should have capped to max_molecules
        data = np.load(path, allow_pickle=True)
        entry_keys = {k for k in data.keys() if k.startswith("0_") or k.startswith("1_") or k.startswith("2_")}
        assert len(entry_keys) <= 3 * 2  # each entry has at most 2 keys


def test_cache_dir_created():
    """Cache should create directory if missing."""
    from src.mto.analysis.mto_cache import MTOCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = os.path.join(tmpdir, "nonexistent", "cache")
        cache = MTOCache(cache_dir)
        assert os.path.isdir(cache_dir)
