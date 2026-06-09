"""Test Stage C forward pass with all spectral tasks (ir, raman, uv)."""
import os, sys, pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="module")
def stage_c_model():
    from src.mto.mto_model import MTONet
    tasks = {"mu": 3, "alpha": 9, "ir": 50, "raman": 50, "uv": 50}
    return MTONet(
        feature_dim=64, mto_hidden_dim=32, readout_hidden_dim=64,
        tasks=tasks, detanet_kwargs={"maxl": 2, "num_block": 2, "rc": 5.0},
        use_activity_gate=True, activity_mode="simple",
    )


def test_stage_c_forward_shape(stage_c_model):
    """Stage C forward should produce all 5 output shapes."""
    n_atoms = 10
    z = torch.randint(1, 10, (n_atoms,))
    pos = torch.randn(n_atoms, 3)
    batch = torch.zeros(n_atoms, dtype=torch.long)

    out = stage_c_model(z=z, pos=pos, batch=batch, return_mto=True)

    assert out["mu"].shape == (1, 3)
    assert out["alpha"].shape == (1, 9)
    assert out["ir"].shape == (1, 50)
    assert out["raman"].shape == (1, 50)
    assert out["uv"].shape == (1, 50)


def test_stage_c_batch_forward(stage_c_model):
    """Multiple molecules in a batch."""
    n_atoms_a = 6
    n_atoms_b = 8
    z = torch.cat([torch.randint(1, 10, (n_atoms_a,)),
                   torch.randint(1, 10, (n_atoms_b,))])
    pos = torch.cat([torch.randn(n_atoms_a, 3), torch.randn(n_atoms_b, 3)])
    batch = torch.cat([torch.zeros(n_atoms_a, dtype=torch.long),
                       torch.ones(n_atoms_b, dtype=torch.long)])

    out = stage_c_model(z=z, pos=pos, batch=batch)
    assert out["mu"].shape == (2, 3)
    assert out["uv"].shape == (2, 50)


def test_stage_c_loss_all_tasks(stage_c_model):
    """Loss with all 5 tasks should be finite."""
    from src.mto.losses import CompositeLoss

    n_atoms = 8
    z = torch.randint(1, 10, (n_atoms,))
    pos = torch.randn(n_atoms, 3)
    batch = torch.zeros(n_atoms, dtype=torch.long)

    criterion = CompositeLoss({
        "mu": 1.0, "alpha": 1.0, "ir": 0.3, "raman": 0.3, "uv": 0.1,
    })
    out = stage_c_model(z=z, pos=pos, batch=batch, return_mto=True)

    targets = {
        "mu": torch.randn(1, 3), "alpha": torch.randn(1, 9),
        "ir": torch.rand(1, 50), "raman": torch.rand(1, 50),
        "uv": torch.rand(1, 50),
    }
    loss, per_task = criterion(out, targets, O=out.get("O"), mask=out.get("mask"))
    assert torch.isfinite(loss)
    assert all(t in per_task for t in ["mu", "alpha", "ir", "raman", "uv"])


def test_stage_c_backward(stage_c_model):
    """Backward pass for Stage C should work."""
    from src.mto.losses import CompositeLoss

    n_atoms = 8
    z = torch.randint(1, 10, (n_atoms,))
    pos = torch.randn(n_atoms, 3)
    batch = torch.zeros(n_atoms, dtype=torch.long)

    criterion = CompositeLoss({
        "mu": 1.0, "alpha": 1.0, "ir": 0.3, "raman": 0.3, "uv": 0.1,
    })
    out = stage_c_model(z=z, pos=pos, batch=batch, return_mto=True)
    targets = {
        "mu": torch.randn(1, 3), "alpha": torch.randn(1, 9),
        "ir": torch.rand(1, 50), "raman": torch.rand(1, 50),
        "uv": torch.rand(1, 50),
    }
    loss, _ = criterion(out, targets, O=out.get("O"), mask=out.get("mask"))
    loss.backward()
    # Check at least readout gradients exist
    assert stage_c_model.readout.heads["mu"].net[0].weight.grad is not None
