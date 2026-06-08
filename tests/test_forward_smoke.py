"""Smoke test for MTO forward pass."""
import torch
import sys, os

# Add project root and third_party paths before any imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)
_detanet_path = os.path.join(_project_root, "third_party", "DetaNet")
if _detanet_path not in sys.path:
    sys.path.insert(0, _detanet_path)

# Import compat at module level
from src.mto.compat import *  # noqa: F401, E402


def test_forward_smoke_synthetic():
    from src.mto.mto_module import ValenceAdaptiveMTO
    C = 128
    mto = ValenceAdaptiveMTO(feature_dim=C)
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    af = torch.randn(9, C)
    batch = torch.zeros(9, dtype=torch.long)
    out = mto(af, z, batch)
    assert "O" in out
    assert "coeff" in out
    assert "mask" in out
    assert "atom_mask" in out
    assert "K_per_mol" in out


def test_forward_with_detanet():
    from src.mto.detanet_adapter import DetaNetBackboneAdapter
    from src.mto.mto_module import ValenceAdaptiveMTO

    backbone = DetaNetBackboneAdapter(num_features=128, maxl=1, num_block=1, rc=5.0)
    backbone.eval()
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    pos = torch.randn(9, 3)
    batch = torch.zeros(9, dtype=torch.long)
    with torch.no_grad():
        bb_out = backbone(z=z, pos=pos, batch=batch)
    af = bb_out["atom_features"]
    assert af.shape == (9, 128)
    mto = ValenceAdaptiveMTO(feature_dim=128)
    out = mto(af, z, bb_out["batch"])
    assert "O" in out
    assert not torch.isnan(out["O"]).any()
