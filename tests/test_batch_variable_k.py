"""Tests for variable-K (valence-adaptive) batch support."""

import torch


def test_variable_k_different_molecules():
    from src.mto.mto_module import ValenceAdaptiveMTO
    
    C = 64
    mto = ValenceAdaptiveMTO(feature_dim=C)
    
    # Ethanol (K=20) + Water (K=8) + Formaldehyde (K=12)
    z = torch.tensor(
        [6, 6, 8, 1, 1, 1, 1, 1, 1,   # ethanol (9 atoms, K=20)
         8, 1, 1,                       # water (3 atoms, K=8)
         6, 8, 1, 1],                   # formaldehyde (4 atoms, K=12)
        dtype=torch.long)
    batch = torch.tensor(
        [0]*9 + [1]*3 + [2]*4, dtype=torch.long)
    
    af = torch.randn(len(z), C)
    out = mto(af, z, batch)
    
    K_per_mol = out["K_per_mol"]
    assert int(K_per_mol[0]) == 20  # ethanol
    assert int(K_per_mol[1]) == 8   # water
    assert int(K_per_mol[2]) == 12  # formaldehyde
    
    K_max = int(K_per_mol.max())
    assert K_max == 20
    
    O = out["O"]
    assert O.shape[0] == 3  # batch size
    assert O.shape[1] == K_max  # K_max
    assert O.shape[2] == C  # feature dim
    
    # Mask should reflect actual K values
    mask = out["mask"]
    assert mask[0, :20].all()
    assert not mask[0, 20:].any()
    assert mask[1, :8].all()
    assert not mask[1, 8:].any()
    assert mask[2, :12].all()
    assert not mask[2, 12:].any()
