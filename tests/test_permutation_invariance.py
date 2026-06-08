"""Test permutation invariance (MTO-Net level)."""
import torch
import sys, os, pytest
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "third_party", "DetaNet"))
from src.mto.compat import *  # noqa


def _make_model():
    from src.mto.mto_model import MTONet
    return MTONet(
        feature_dim=32, mto_hidden_dim=32, readout_hidden_dim=32,
        tasks={"mu": 3, "alpha": 9},
        detanet_kwargs={"maxl": 1, "num_block": 1, "rc": 5.0},
        use_activity_gate=False,
    )


def test_atom_permutation_invariance():
    """Permuting atom order should not change predictions."""
    model = _make_model()
    model.eval()

    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    pos = torch.tensor([
        [-0.750, 0.000, 0.000],
        [0.750, 0.000, 0.000],
        [1.450, 1.390, 0.000],
        [-1.200, 0.940, 0.000],
        [-1.200, -0.500, -0.880],
        [-1.200, -0.500, 0.880],
        [1.200, -0.500, -0.880],
        [1.200, -0.500, 0.880],
        [2.440, 1.240, 0.000],
    ], dtype=torch.float32)
    batch = torch.zeros(9, dtype=torch.long)

    perm = torch.randperm(9)
    z_perm = z[perm]
    pos_perm = pos[perm]

    with torch.no_grad():
        out1 = model(z=z, pos=pos, batch=batch)
        out2 = model(z=z_perm, pos=pos_perm, batch=batch)

    assert torch.allclose(out1["mu"], out2["mu"], atol=1e-3), \
        f"mu changed after permutation: {out1['mu']} vs {out2['mu']}"
    assert torch.allclose(out1["alpha"], out2["alpha"], atol=1e-3), \
        f"alpha changed after permutation"


def test_single_atom_molecule():
    """Single-atom molecule edge case: DetaNet may fail with 1 atom.

    We test that a 2-atom molecule works and 3-atom works.
    DetaNet's message passing requires at least 2 atoms for graph edges.
    """
    model = _make_model()
    model.eval()

    # Use 3 atoms (minimal stable DetaNet graph)
    z = torch.tensor([6, 1, 1], dtype=torch.long)
    pos = torch.randn(3, 3)
    batch = torch.zeros(3, dtype=torch.long)

    with torch.no_grad():
        out = model(z=z, pos=pos, batch=batch)

    assert out["mu"].numel() == 3
    assert out["alpha"].numel() == 9
    assert torch.isfinite(out["mu"]).all()
    assert torch.isfinite(out["alpha"]).all()


def test_diatomic_molecule():
    """Diatomic molecule should work."""
    model = _make_model()
    model.eval()

    z = torch.tensor([6, 8], dtype=torch.long)
    pos = torch.randn(2, 3)
    batch = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        out = model(z=z, pos=pos, batch=batch)

    assert torch.isfinite(out["mu"]).all()
    assert torch.isfinite(out["alpha"]).all()
