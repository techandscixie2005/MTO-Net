"""Tests for stability analysis metrics."""
import numpy as np

def test_subspace_similarity_identical():
    from src.mto.stability import subspace_similarity
    r = 5
    d = 20
    Q = np.random.randn(r, d)
    sim = subspace_similarity(Q, Q, r)
    assert 0.85 < sim <= 1.01, f"Expected ~1.0, got {sim}"

def test_subspace_similarity_dimensions():
    from src.mto.stability import subspace_similarity
    r = 3
    d = 10
    Q = np.random.randn(r, d)
    sim = subspace_similarity(Q, Q, r)
    assert 0.85 < sim

def test_seed_subspace_stability():
    from src.mto.stability import seed_subspace_stability
    contrib_maps = {
        "seed_0": {"contributions": {0: np.random.randn(5, 20), 1: np.random.randn(5, 15)}},
        "seed_1": {"contributions": {0: np.random.randn(5, 20), 1: np.random.randn(5, 15)}},
    }
    result = seed_subspace_stability(contrib_maps, top_r=5)
    assert "pairs" in result
    assert len(result["pairs"]) == 1

def test_stage_stability():
    from src.mto.stability import stage_stability
    contribs_a = {"contributions": {0: np.random.randn(5, 20), 1: np.random.randn(5, 15)}}
    contribs_b = {"contributions": {0: np.random.randn(5, 20), 1: np.random.randn(5, 15)}}
    result = stage_stability(contribs_a, contribs_b)
    assert "mean_correlation" in result
    assert "n_mols" in result
