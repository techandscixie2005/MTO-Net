"""Test Stage B forward pass with spectral tasks (ir, raman)."""
import os, sys, pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="module")
def stage_b_model():
    from src.mto.mto_model import MTONet
    tasks = {"mu": 3, "alpha": 9, "ir": 100, "raman": 100}
    return MTONet(
        feature_dim=64, mto_hidden_dim=32, readout_hidden_dim=64,
        tasks=tasks, detanet_kwargs={"maxl": 2, "num_block": 2, "rc": 5.0},
        use_activity_gate=True, activity_mode="simple",
    )


def test_stage_b_forward_shape(stage_b_model):
    """Stage B forward should produce correct output shapes."""
    n_atoms = 8
    z = torch.randint(1, 10, (n_atoms,))
    pos = torch.randn(n_atoms, 3)
    batch = torch.zeros(n_atoms, dtype=torch.long)

    out = stage_b_model(z=z, pos=pos, batch=batch, return_mto=True)

    assert out["mu"].shape == (1, 3)
    assert out["alpha"].shape == (1, 9)
    assert out["ir"].shape == (1, 100)
    assert out["raman"].shape == (1, 100)
    assert out["O"].ndim == 3  # [B, K, C]
    assert out["mask"].dtype == torch.bool


def test_stage_b_loss_finite(stage_b_model):
    """Loss should be finite on a forward pass."""
    from src.mto.losses import CompositeLoss

    n_atoms = 8
    z = torch.randint(1, 10, (n_atoms,))
    pos = torch.randn(n_atoms, 3)
    batch = torch.zeros(n_atoms, dtype=torch.long)

    criterion = CompositeLoss({"mu": 1.0, "alpha": 1.0, "ir": 0.3, "raman": 0.3})
    out = stage_b_model(z=z, pos=pos, batch=batch, return_mto=True)

    targets = {
        "mu": torch.randn(1, 3),
        "alpha": torch.randn(1, 9),
        "ir": torch.rand(1, 100),
        "raman": torch.rand(1, 100),
    }

    loss, per_task = criterion(
        out, targets,
        O=out.get("O"),
        mask=out.get("mask"),
    )
    assert torch.isfinite(loss)
    assert "mu" in per_task
    assert "ir" in per_task


def test_stage_b_backward(stage_b_model):
    """Backward pass should work without errors."""
    n_atoms = 8
    z = torch.randint(1, 10, (n_atoms,))
    pos = torch.randn(n_atoms, 3)
    batch = torch.zeros(n_atoms, dtype=torch.long)

    from src.mto.losses import CompositeLoss
    criterion = CompositeLoss({"mu": 1.0, "alpha": 1.0, "ir": 0.3, "raman": 0.3})
    out = stage_b_model(z=z, pos=pos, batch=batch, return_mto=True)

    targets = {
        "mu": torch.randn(1, 3),
        "alpha": torch.randn(1, 9),
        "ir": torch.rand(1, 100),
        "raman": torch.rand(1, 100),
    }
    loss, _ = criterion(out, targets, O=out.get("O"), mask=out.get("mask"))
    loss.backward()

    # Check that at least the readout and MTO module have gradients
    grad_params = sum(1 for p in stage_b_model.parameters()
                     if p.requires_grad and p.grad is not None)
    total_params = sum(1 for p in stage_b_model.parameters() if p.requires_grad)
    assert grad_params >= total_params * 0.5, (
        f"Only {grad_params}/{total_params} params have grads - "
        f"expected most params to receive gradients"
    )
    # Verify key components receive gradients
    assert stage_b_model.readout.heads["mu"].net[0].weight.grad is not None
    assert stage_b_model.readout.heads["ir"].net[0].weight.grad is not None
