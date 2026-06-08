"""Test dataset split locking.

Ensures split is generated, saved, reloaded, and reused deterministically.
"""
import os
import sys
import json
import tempfile
import random as std_random

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)


def test_generate_split_sizes():
    from src.mto.data_splits import generate_split
    split = generate_split(100, train_frac=0.8, val_frac=0.1, seed=0)
    assert len(split["train"]) == 80
    assert len(split["val"]) == 10
    assert len(split["test"]) == 10
    assert split["split_seed"] == 0
    assert split["n_total"] == 100


def test_split_reproducibility():
    from src.mto.data_splits import generate_split
    s1 = generate_split(100, seed=42)
    s2 = generate_split(100, seed=42)
    assert s1["train"] == s2["train"]
    assert s1["val"] == s2["val"]
    assert s1["test"] == s2["test"]


def test_split_different_seeds_different_order():
    """Different split seeds should produce different train sets."""
    from src.mto.data_splits import generate_split
    s1 = generate_split(100, seed=0)
    s2 = generate_split(100, seed=1)
    # They should differ (very unlikely to be identical with 100 items)
    assert s1["train"] != s2["train"]


def test_save_load_split():
    from src.mto.data_splits import generate_split, save_split, load_split
    split = generate_split(100, seed=7)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_split.json")
    try:
        h = save_split(split, path)
        assert os.path.exists(path)
        assert os.path.exists(path.replace(".json", "_hash.txt"))
        loaded = load_split(path)
        assert loaded["train"] == split["train"]
        assert loaded["val"] == split["val"]
        assert loaded["test"] == split["test"]
        assert loaded["split_seed"] == split["split_seed"]
    finally:
        os.remove(path)
        os.remove(path.replace(".json", "_hash.txt"))
        os.rmdir(tmpdir)


def test_load_or_create():
    from src.mto.data_splits import load_or_create_split
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_split2.json")
    try:
        # First call creates
        split1, created = load_or_create_split(path, 100, seed=3)
        assert created is True
        # Second call loads
        split2, created = load_or_create_split(path, 100, seed=99)
        assert created is False
        # Should be same split (seed=3 from first call)
        assert split1["train"] == split2["train"]
    finally:
        if os.path.exists(path):
            os.remove(path)
        hp = path.replace(".json", "_hash.txt")
        if os.path.exists(hp):
            os.remove(hp)
        os.rmdir(tmpdir)


def test_seed_does_not_change_split():
    """Training seed should not affect dataset membership."""
    from src.mto.data_splits import load_or_create_split, split_indices_for_seed
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_split3.json")
    try:
        split, _ = load_or_create_split(path, 100, seed=0)
        # Different training seeds should return same indices
        t0, v0, ts0 = split_indices_for_seed(split, 0)
        t1, v1, ts1 = split_indices_for_seed(split, 1)
        t99, v99, ts99 = split_indices_for_seed(split, 99)
        assert t0 == t1 == t99
        assert v0 == v1 == v99
        assert ts0 == ts1 == ts99
    finally:
        if os.path.exists(path):
            os.remove(path)
        hp = path.replace(".json", "_hash.txt")
        if os.path.exists(hp):
            os.remove(hp)
        os.rmdir(tmpdir)


def test_incompatible_split_sizes():
    from src.mto.data_splits import load_or_create_split
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_split4.json")
    try:
        split1, _ = load_or_create_split(path, 100, seed=0)
        import pytest
        with pytest.raises(ValueError):
            load_or_create_split(path, 200, seed=0)
    finally:
        if os.path.exists(path):
            os.remove(path)
        hp = path.replace(".json", "_hash.txt")
        if os.path.exists(hp):
            os.remove(hp)
        os.rmdir(tmpdir)
