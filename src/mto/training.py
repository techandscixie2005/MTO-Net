"""Training utilities for MTO-Net."""

import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import CompositeLoss, default_task_weights


class NormalizationStats:
    """Gathers and stores per-property mean/std for standardization."""

    def __init__(self):
        self.stats = {}

    def fit_tensors(self, key: str, tensors: list[torch.Tensor]):
        all_vals = torch.cat([t.reshape(-1) for t in tensors])
        self.stats[key] = {
            "mean": float(all_vals.mean().item()),
            "std": float(all_vals.std().item()) if all_vals.std().item() > 1e-8 else 1.0,
        }

    def normalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        s = self.stats[key]
        return (tensor - s["mean"]) / s["std"]

    def denormalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        s = self.stats[key]
        return tensor * s["std"] + s["mean"]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        with open(path) as f:
            obj.stats = json.load(f)
        return obj


class Trainer:
    """Simple training loop for MTO-Net."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        tasks: list[str],
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        task_weights: dict | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.tasks = tasks
        self.criterion = CompositeLoss(task_weights or default_task_weights(tasks))
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

    def train_epoch(self, loader: DataLoader, norm_stats: NormalizationStats = None) -> dict:
        self.model.train()
        total_loss = 0.0
        n = 0
        task_losses = {}

        for batch in loader:
            self.optimizer.zero_grad()
            preds, targets = self._forward(batch, norm_stats)
            loss, per_task = self.criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n += 1
            for k, v in per_task.items():
                task_losses[k] = task_losses.get(k, 0.0) + v

        return {
            "loss": total_loss / n,
            "task_losses": {k: v / n for k, v in task_losses.items()},
        }

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader, norm_stats: NormalizationStats = None) -> dict:
        self.model.eval()
        total_loss = 0.0
        n = 0
        task_losses = {}

        for batch in loader:
            preds, targets = self._forward(batch, norm_stats)
            loss, per_task = self.criterion(preds, targets)
            total_loss += loss.item()
            n += 1
            for k, v in per_task.items():
                task_losses[k] = task_losses.get(k, 0.0) + v

        return {
            "loss": total_loss / n,
            "task_losses": {k: v / n for k, v in task_losses.items()},
        }

    def _forward(self, batch, norm_stats):
        z = batch["z"].to(self.device)
        pos = batch["pos"].to(self.device)
        batch_idx = batch.get("batch", torch.zeros(len(z), dtype=torch.long)).to(self.device)

        out = self.model(z=z, pos=pos, batch=batch_idx)

        preds = {}
        targets = {}
        for task in self.tasks:
            if task in out and task in batch:
                targets[task] = batch[task].to(self.device)
                preds[task] = out[task]

        if norm_stats is not None:
            targets = {k: norm_stats.normalize(k, v) for k, v in targets.items()}

        return preds, targets

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        norm_stats: NormalizationStats = None,
        checkpoint_dir: str = None,
        log_interval: int = 10,
    ):
        best_val_loss = float("inf")
        best_path = None
        history = {"train": [], "val": []}

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, norm_stats)
            val_metrics = self.eval_epoch(val_loader, norm_stats)
            self.scheduler.step(val_metrics["loss"])

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            if val_metrics["loss"] < best_val_loss and checkpoint_dir:
                best_val_loss = val_metrics["loss"]
                best_path = os.path.join(checkpoint_dir, "best.pt")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "history": history,
                    },
                    best_path,
                )

            if epoch % log_interval == 0:
                print(f"Epoch {epoch:4d} | train loss: {train_metrics[loss]:.4f} | "
                      f"val loss: {val_metrics[loss]:.4f}")

        return history, best_path


def compute_metrics(preds: dict, targets: dict) -> dict:
    """Compute MAE, RMSE, R2, cosine similarity per task."""
    import numpy as np

    metrics = {}
    for task in preds:
        p = preds[task].detach().cpu().numpy()
        t = targets[task].detach().cpu().numpy()
        mae = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        metrics[task] = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
    return metrics
