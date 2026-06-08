"""Tests for MTO module output shapes."""
import torch

def test_mto_output_shapes():
    from src.mto.mto_module import ValenceAdaptiveMTO
    C = 128
    mto = ValenceAdaptiveMTO(feature_dim=C)
    N = 20
    # Use only valid atomic numbers: H=1, C=6, N=7, O=8, F=9
    valid_z = torch.tensor([1, 6, 7, 8, 9])
    z = valid_z[torch.randint(0, 5, (N,))]
    af = torch.randn(N, C)
    batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    out = mto(af, z, batch)
    B = 2
    Kp = int(out["K_per_mol"].max())
    assert out["O"].shape == (B, Kp, C)
    assert out["coeff"].shape[0] == B
    assert out["coeff"].shape[1] == Kp
    assert out["mask"].shape == (B, Kp)
    assert out["atom_mask"].shape[0] == B

def test_mto_no_nan():
    from src.mto.mto_module import ValenceAdaptiveMTO
    C = 128
    mto = ValenceAdaptiveMTO(feature_dim=C)
    N = 15
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1, 6, 1, 1], dtype=torch.long)
    af = torch.randn(N, C)
    batch = torch.tensor([0]*9 + [1]*6, dtype=torch.long)
    out = mto(af, z, batch)
    assert not torch.isnan(out["O"]).any()
    assert not torch.isnan(out["coeff"]).any()

def test_mto_coeff_normalized():
    from src.mto.mto_module import ValenceAdaptiveMTO
    C = 64
    mto = ValenceAdaptiveMTO(feature_dim=C)
    z = torch.tensor([6, 8, 1, 1], dtype=torch.long)
    af = torch.randn(4, C)
    batch = torch.zeros(4, dtype=torch.long)
    out = mto(af, z, batch)
    coeff = out["coeff_flat_list"][0]
    abs_sum = coeff.abs().sum(dim=-1)
    assert torch.allclose(abs_sum, torch.ones_like(abs_sum), atol=1e-5)
