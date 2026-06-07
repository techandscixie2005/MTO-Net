"""Loss functions for MTO-Net properties."""
import torch, torch.nn as nn, torch.nn.functional as F

class DipoleMSELoss(nn.Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)

class AlphaLoss(nn.Module):
    """MSE on flattened 3x3 polarizability."""
    def forward(self, pred, target):
        return F.mse_loss(pred, target.reshape(pred.shape))

class SpectralLoss(nn.Module):
    def __init__(self, cosine_weight=0.1):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        mse = self.mse(pred, target)
        cos = F.cosine_similarity(pred, target, dim=-1).mean()
        return mse - self.cosine_weight * cos

class CompositeLoss(nn.Module):
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or {}
        self.dipole_loss = DipoleMSELoss()
        self.alpha_loss = AlphaLoss()
        self.spectral_loss = SpectralLoss()
    def forward(self, preds, targets):
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

def default_task_weights(tasks):
    weights = {"mu": 1.0, "alpha": 1.0, "ir": 0.3, "raman": 0.3, "uv": 0.1}
    return {k: weights.get(k, 1.0) for k in tasks}
