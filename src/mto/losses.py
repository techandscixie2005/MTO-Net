"""Loss functions for MTO-Net properties."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DipoleMSELoss(nn.Module):
    """MSE loss for dipole moment vectors (standardized)."""

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class PolarizabilityLoss(nn.Module):
    """Frobenius MSE loss for polarizability tensors.

    pred: [B, 6] (upper triangular) or [B, 3, 3]
    target: same format
    """

    def forward(self, pred, target):
        if pred.dim() == 2 and pred.shape[-1] == 6:
            pred = _tri_to_mat(pred)
        if target.dim() == 2 and target.shape[-1] == 6:
            target = _tri_to_mat(target)
        return F.mse_loss(pred, target)


def _tri_to_mat(tri):
    """Convert [B, 6] upper triangular to [B, 3, 3]."""
    B = tri.shape[0]
    mat = torch.zeros(B, 3, 3, device=tri.device, dtype=tri.dtype)
    mat[:, 0, 0] = tri[:, 0]
    mat[:, 0, 1] = tri[:, 1]
    mat[:, 0, 2] = tri[:, 3]
    mat[:, 1, 0] = tri[:, 1]
    mat[:, 1, 1] = tri[:, 2]
    mat[:, 1, 2] = tri[:, 4]
    mat[:, 2, 0] = tri[:, 3]
    mat[:, 2, 1] = tri[:, 4]
    mat[:, 2, 2] = tri[:, 5]
    return mat


class SpectralLoss(nn.Module):
    """MSE + optional cosine similarity loss for spectra."""

    def __init__(self, cosine_weight: float = 0.1):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        if self.cosine_weight > 0:
            cos = F.cosine_similarity(pred, target, dim=-1).mean()
            return mse - self.cosine_weight * cos
        return mse


class CompositeLoss(nn.Module):
    """Weighted composite loss for multi-task training.

    task_weights: dict task_name -> float weight
    """

    def __init__(self, task_weights: dict | None = None):
        super().__init__()
        self.task_weights = task_weights or {}
        self.dipole_loss = DipoleMSELoss()
        self.alpha_loss = PolarizabilityLoss()
        self.spectral_loss = SpectralLoss()

    def forward(self, preds: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        """Returns (total_loss, per_task_loss_dict)."""
        total = 0.0
        per_task = {}

        for name in preds:
            if name not in targets:
                continue
            pred = preds[name]
            target = targets[name]
            w = self.task_weights.get(name, 1.0)

            if name == "mu":
                loss = self.dipole_loss(pred, target) * w
            elif name == "alpha":
                loss = self.alpha_loss(pred, target) * w
            elif name in ("ir", "raman", "uv"):
                loss = self.spectral_loss(pred, target) * w
            else:
                loss = F.mse_loss(pred, target) * w

            total = total + loss
            per_task[name] = loss.detach().item()

        return total, per_task


def default_task_weights(tasks: list[str]) -> dict:
    """Default balanced weights per task."""
    weights = {
        "mu": 1.0,
        "alpha": 1.0,
        "ir": 0.3,
        "raman": 0.3,
        "uv": 0.1,
    }
    return {k: weights.get(k, 1.0) for k in tasks}
