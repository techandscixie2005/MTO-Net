"""Loss functions for MTO-Net including diversity and entropy regularization.

Per TOTAL.md 11-12: task-specific losses, diversity loss, routing entropy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, task_weights=None, diversity_weight=1e-3, entropy_weight=1e-3):
        super().__init__()
        self.task_weights = task_weights or {}
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
        self.dipole_loss = DipoleMSELoss()
        self.alpha_loss = AlphaLoss()
        self.spectral_loss = SpectralLoss()

    def forward(self, preds, targets, O=None, mask=None, routing_logits=None):
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

        # Diversity loss: discourage MTO slot collapse (TOTAL 12.1)
        if O is not None and mask is not None and self.diversity_weight > 0:
            div_loss = self._diversity_loss(O, mask)
            total = total + self.diversity_weight * div_loss
            per_task["diversity"] = div_loss.detach().item()

        # Routing entropy: early mild encouragement (TOTAL 12.2)
        if routing_logits is not None and self.entropy_weight > 0 and len(routing_logits) > 0:
            ent_loss = self._routing_entropy(routing_logits)
            total = total + self.entropy_weight * ent_loss
            per_task["entropy"] = ent_loss.detach().item()

        return total, per_task

    def _diversity_loss(self, O, mask):
        B, K, C = O.shape
        total_div = 0.0
        count = 0
        for b in range(B):
            valid = mask[b]
            k_valid = valid.sum().item()
            if k_valid < 2:
                continue
            O_valid = O[b, valid]
            O_norm = F.normalize(O_valid, p=2, dim=-1)
            G = O_norm @ O_norm.T
            off_diag = G - torch.eye(k_valid, device=G.device)
            total_div = total_div + (off_diag ** 2).sum()
            count = count + 1
        if count > 0:
            return total_div / count
        return torch.tensor(0.0, device=O.device)

    def _routing_entropy(self, routing_logits):
        total_ent = 0.0
        count = 0
        for logits in routing_logits:
            if logits.numel() == 0:
                continue
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
            total_ent = total_ent + (-ent)
            count = count + 1
        if count > 0:
            return total_ent / count
        return torch.tensor(0.0, device=logits.device)


def default_task_weights(tasks):
    weights = {"mu": 1.0, "alpha": 1.0, "ir": 0.3, "raman": 0.3, "uv": 0.1}
    return {k: weights.get(k, 1.0) for k in tasks}
