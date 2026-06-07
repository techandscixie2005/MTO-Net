"""Training utilities for MTO-Net."""
import json, os, time
from typing import Optional
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from .losses import CompositeLoss, default_task_weights

class NormalizationStats:
    """Per-property mean/std for standardization."""
    def __init__(self):
        self.stats = {}
    def fit_tensors(self, key, tensors):
        all_vals = torch.cat([t.reshape(-1) for t in tensors])
        s = float(all_vals.std().item())
        self.stats[key] = {"mean": float(all_vals.mean().item()), "std": s if s > 1e-8 else 1.0}
    def normalize(self, key, tensor):
        s = self.stats[key]
        return (tensor - s["mean"]) / s["std"]
    def denormalize(self, key, tensor):
        s = self.stats[key]
        return tensor * s["std"] + s["mean"]
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)
    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path) as f:
            obj.stats = json.load(f)
        return obj

class Trainer:
    def __init__(self, model, device, tasks, lr=1e-3, weight_decay=1e-5, task_weights=None):
        self.model = model.to(device)
        self.device = device
        self.tasks = tasks
        self.criterion = CompositeLoss(task_weights or default_task_weights(tasks))
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)

    def train_epoch(self, loader, norm_stats=None):
        self.model.train()
        total_loss, n, task_losses = 0.0, 0, {}
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
        return {"loss": total_loss / n, "task_losses": {k: v / n for k, v in task_losses.items()}}

    @torch.no_grad()
    def eval_epoch(self, loader, norm_stats=None):
        self.model.eval()
        total_loss, n, task_losses = 0.0, 0, {}
        for batch in loader:
            preds, targets = self._forward(batch, norm_stats)
            loss, per_task = self.criterion(preds, targets)
            total_loss += loss.item()
            n += 1
            for k, v in per_task.items():
                task_losses[k] = task_losses.get(k, 0.0) + v
        return {"loss": total_loss / n, "task_losses": {k: v / n for k, v in task_losses.items()}}

    def _forward(self, batch, norm_stats):
        z = batch["z"].to(self.device)
        pos = batch["pos"].to(self.device)
        batch_idx = batch.get("batch", torch.zeros(len(z), dtype=torch.long)).to(self.device)
        out = self.model(z=z, pos=pos, batch=batch_idx)
        preds, targets = {}, {}
        for task in self.tasks:
            if task in out and task in batch:
                targets[task] = batch[task].to(self.device)
                preds[task] = out[task]
        if norm_stats is not None:
            targets = {k: norm_stats.normalize(k, v) for k, v in targets.items()}
        return preds, targets

    def fit(self, train_loader, val_loader, epochs, norm_stats=None, checkpoint_dir=None, log_interval=10):
        best_val_loss = float("inf")
        best_path = None
        history = {"train": [], "val": []}
        for epoch in range(epochs):
            train_m = self.train_epoch(train_loader, norm_stats)
            val_m = self.eval_epoch(val_loader, norm_stats)
            self.scheduler.step(val_m["loss"])
            history["train"].append(train_m)
            history["val"].append(val_m)

            # Save last checkpoint every epoch
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                last_path = os.path.join(checkpoint_dir, "last.pt")
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "epoch": epoch, "val_loss": val_m["loss"], "history": history,
                }, last_path)

            if val_m["loss"] < best_val_loss and checkpoint_dir:
                best_val_loss = val_m["loss"]
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "epoch": epoch, "best_epoch": epoch,
                    "best_val_loss": best_val_loss, "val_metrics": val_m,
                    "train_loss": train_m["loss"], "history": history,
                }, best_path)

            if epoch % log_interval == 0:
                print("Epoch {:4d} | train loss: {:.4f} | val loss: {:.4f}".format(epoch, train_m["loss"], val_m["loss"]))

        return history, best_path


def compute_metrics(preds, targets):
    import numpy as np
    metrics = {}
    for task in preds:
        p = preds[task].detach().cpu().numpy()
        t = targets[task].detach().cpu().numpy()
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-10))
        metrics[task] = {"mae": mae, "rmse": rmse, "r2": r2}
    return metrics
