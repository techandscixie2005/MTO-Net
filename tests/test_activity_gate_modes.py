"""Test all three activity gate modes: none, simple, fermi_dirac."""
import sys, os
import torch
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _project_root)


def test_gate_none():
    from src.mto.activity_gate import ActivityGate
    B, K, C = 2, 20, 64
    gate = ActivityGate(C, hidden_dim=32, mode="none")
    O = torch.randn(B, K, C)
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[0, :14] = True
    mask[1, :8] = True
    n, O_active, eps, mu = gate(O, mask)
    # All valid slots should be active (n_k = 1.0)
    assert torch.allclose(n[0, mask[0]], torch.ones(14))
    assert torch.allclose(n[1, mask[1]], torch.ones(8))
    # Padded slots should be 0
    assert (n[0, ~mask[0]] == 0).all()
    assert eps is None


def test_gate_simple():
    from src.mto.activity_gate import ActivityGate
    B, K, C = 2, 20, 64
    gate = ActivityGate(C, hidden_dim=32, mode="simple")
    O = torch.randn(B, K, C)
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[0, :14] = True
    mask[1, :8] = True
    n, O_active, eps, mu = gate(O, mask)
    assert n.shape == (B, K)
    assert O_active.shape == (B, K, C)
    # Values in [0, 1]
    assert (n[0, mask[0]] >= 0).all() and (n[0, mask[0]] <= 1).all()
    # Padded slots are 0
    assert (n[0, ~mask[0]] == 0).all()
    # Differentiable
    loss = n.sum()
    loss.backward()


def test_gate_fermi_dirac():
    from src.mto.activity_gate import ActivityGate
    B, K, C = 2, 20, 64
    gate = ActivityGate(C, hidden_dim=32, mode="fermi_dirac")
    O = torch.randn(B, K, C)
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[0, :14] = True
    mask[1, :8] = True
    N_val = torch.tensor([14.0, 8.0])
    n, O_active, eps, mu = gate(O, mask, N_val, theta=0.5)
    assert n.shape == (B, K)
    # Charge conservation
    assert abs(n[0, mask[0]].sum().item() - 14.0) < 0.1
    assert abs(n[1, mask[1]].sum().item() - 8.0) < 0.1
    # Padded slots 0
    assert (n[0, ~mask[0]] == 0).all()
    # Differentiable
    loss = n.sum()
    loss.backward()


def test_all_modes_respect_mask():
    from src.mto.activity_gate import ActivityGate
    B, K, C = 2, 10, 32
    O = torch.randn(B, K, C)
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[0, :5] = True
    mask[1, :3] = True

    for mode in ["none", "simple", "fermi_dirac"]:
        gate = ActivityGate(C, hidden_dim=32, mode=mode)
        kwargs = {}
        if mode == "fermi_dirac":
            kwargs["N_val"] = torch.tensor([5.0, 3.0])
        n, _, _, _ = gate(O, mask, **kwargs)
        assert (n[0, 5:] == 0).all(), f"mode={mode}: padded mol 0 should be zero"
        assert (n[1, 3:] == 0).all(), f"mode={mode}: padded mol 1 should be zero"
        assert n[0, :5].sum() > 0, f"mode={mode}: valid slots should be active"


def test_no_mode_uses_charge():
    """None of the modes should use or require molecular charge Q."""
    from src.mto.activity_gate import ActivityGate
    C = 32
    O = torch.randn(2, 10, C)
    mask = torch.ones(2, 10, dtype=torch.bool)
    # none mode - no N_val needed
    ActivityGate(C, mode="none")(O, mask)
    # simple mode - no N_val needed
    ActivityGate(C, mode="simple")(O, mask)
    # fermi_dirac - uses N_val (valence electrons), NOT charge Q
    gate_fd = ActivityGate(C, mode="fermi_dirac")
    gate_fd(O, mask, N_val=torch.tensor([10.0, 10.0]))

    # Verify ActivityGate signature has no Q parameter
    import inspect
    sig = inspect.signature(gate_fd.forward)
    params = list(sig.parameters.keys())
    assert "Q" not in params, "forward() should not have a Q (charge) parameter"
    assert "charge" not in params, "forward() should not have a charge parameter"
