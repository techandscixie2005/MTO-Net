"""Test spectral loss functions: MSE, cosine similarity, multi-task weighting."""
import os, sys, pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_spectral_mse_loss():
    from src.mto.losses import SpectralLoss
    loss_fn = SpectralLoss(cosine_weight=0.0)
    pred = torch.randn(4, 100)
    target = torch.randn(4, 100)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    assert loss > 0


def test_spectral_cosine_loss():
    from src.mto.losses import SpectralLoss
    loss_fn = SpectralLoss(cosine_weight=0.5)
    pred = torch.randn(4, 100)
    target = torch.randn(4, 100)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)


def test_spectral_perfect_match():
    from src.mto.losses import SpectralLoss
    loss_fn = SpectralLoss(cosine_weight=0.0)
    pred = torch.ones(4, 100)
    target = torch.ones(4, 100)
    loss = loss_fn(pred, target)
    assert loss.item() < 1e-5


def test_composite_loss_spectral():
    from src.mto.losses import CompositeLoss
    criterion = CompositeLoss({
        "mu": 1.0, "ir": 0.3, "raman": 0.3,
    })
    preds = {
        "mu": torch.randn(2, 3),
        "ir": torch.randn(2, 100),
        "raman": torch.randn(2, 100),
    }
    targets = {
        "mu": torch.randn(2, 3),
        "ir": torch.randn(2, 100),
        "raman": torch.randn(2, 100),
    }
    loss, per_task = criterion(preds, targets)
    assert torch.isfinite(loss)
    assert "mu" in per_task
    assert "ir" in per_task
    assert "raman" in per_task


def test_missing_task_skipped():
    from src.mto.losses import CompositeLoss
    criterion = CompositeLoss({"mu": 1.0, "ir": 0.3})
    preds = {"mu": torch.randn(2, 3)}  # ir missing
    targets = {"mu": torch.randn(2, 3), "ir": torch.randn(2, 100)}
    loss, per_task = criterion(preds, targets)
    assert torch.isfinite(loss)
    assert "ir" not in per_task  # skipped because pred missing


def test_default_task_weights():
    from src.mto.losses import default_task_weights
    weights = default_task_weights(["mu", "alpha", "ir", "raman", "uv"])
    assert weights["mu"] == 1.0
    assert weights["ir"] == 0.3
    assert weights["uv"] == 0.1


def test_normalization_stats_spectral():
    from src.mto.training import NormalizationStats
    stats = NormalizationStats()
    tensors = [torch.randn(100) * 10 + 5 for _ in range(50)]
    stats.fit_tensors("ir", tensors)
    s = stats.stats["ir"]
    assert abs(s["mean"] - 5.0) < 2.0
    assert abs(s["std"] - 10.0) < 3.0

    # Normalize and denormalize
    t = torch.tensor([15.0, 5.0, -5.0])
    normed = stats.normalize("ir", t)
    recovered = stats.denormalize("ir", normed)
    assert torch.allclose(recovered, t, atol=1e-4)
