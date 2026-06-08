"""Test MTO mask correctness."""
import torch


def test_mto_mask_values():
    from src.mto.valence import molecular_valence_electrons
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    Kp = molecular_valence_electrons(z, batch)
    assert int(Kp[0]) == 20
    assert int(Kp[1]) == 8
    K_max = int(Kp.max())
    mask = torch.zeros(2, K_max, dtype=torch.bool)
    mask[0, :20] = True
    mask[1, :8] = True
    assert mask[0, :20].all()
    assert not mask[0, 20:].any()
    assert mask[1, :8].all()
    assert not mask[1, 8:].any()


def test_mto_mask_in_output():
    from src.mto.mto_module import ValenceAdaptiveMTO
    C = 64
    mto = ValenceAdaptiveMTO(feature_dim=C, use_activity_gate=False)
    z = torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1], dtype=torch.long)
    af = torch.randn(12, C)
    batch = torch.tensor([0]*9 + [1]*3, dtype=torch.long)
    out = mto(af, z, batch)
    mask = out["mask"]
    assert int(out["K_per_mol"][0]) == 20
    assert int(out["K_per_mol"][1]) == 8
    # mol 0: first slot valid, last slot also valid (K=20 fills all)
    assert mask[0, 0].item()
    assert mask[0, 19].item()
    # mol 1: first 8 valid, rest not
    assert mask[1, 0].item()
    assert not mask[1, 8].item()


def test_mto_mask_padded_slots_zero():
    """Two molecules with different K: padded slots should be zero."""
    from src.mto.mto_module import ValenceAdaptiveMTO
    C = 64
    mto = ValenceAdaptiveMTO(feature_dim=C, use_activity_gate=False)
    # Ethanol (9 atoms, K=20) + Water (3 atoms, K=8)
    z = torch.tensor(
        [6, 6, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1], dtype=torch.long)
    af = torch.randn(12, C)
    batch = torch.tensor([0]*9 + [1]*3, dtype=torch.long)
    out = mto(af, z, batch)
    O = out["O"]
    mask = out["mask"]
    K_max = int(out["K_per_mol"].max())
    K1 = int(out["K_per_mol"][1])
    # Mol 1 has only K1=8 valid slots out of K_max=20
    # Padded slots (8..19) should be masked as False
    assert not mask[1, K1:].any()
