"""Training utilities for MTO-Net."""
import json, os, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from .losses import CompositeLoss, default_task_weights

class NormalizationStats:
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10)

    def train_epoch(self, loader, norm_stats=None):
        self.model.train()
        total_loss, n = 0.0, 0
        task_losses = {}
        for batch in loader:
            self.optimizer.zero_grad()
            batch_loss, per_task = self._process_batch(batch, norm_stats)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += batch_loss.item()
            n += 1
            for k, v in per_task.items():
                task_losses[k] = task_losses.get(k, 0.0) + v
        return {"loss": total_loss / n, "task_losses": {k: v / n for k, v in task_losses.items()}}

    @torch.no_grad()
    def eval_epoch(self, loader, norm_stats=None):
        self.model.eval()
        total_loss, n, task_losses = 0.0, 0, {}
        for batch in loader:
            batch_loss, per_task = self._process_batch(batch, norm_stats)
            total_loss += batch_loss.item()
            n += 1
            for k, v in per_task.items():
                task_losses[k] = task_losses.get(k, 0.0) + v
        return {"loss": total_loss / n, "task_losses": {k: v / n for k, v in task_losses.items()}}

    def _process_batch(self, batch, norm_stats):
        """Process a batch of molecules (one sample per batch item)."""
        total_loss = 0.0
        per_task = {}
        B = len(batch["z"]) if isinstance(batch["z"], list) else batch["z"].shape[0]

        for i in range(B):
            if isinstance(batch["z"], list):
                z_i = batch["z"][i].to(self.device)
                pos_i = batch["pos"][i].to(self.device)
                batch_i = torch.zeros(len(z_i), dtype=torch.long, device=self.device)
            else:
                z_i = batch["z"][i].to(self.device)
                pos_i = batch["pos"][i].to(self.device)
                batch_i = batch.get("batch", torch.zeros(len(z_i), dtype=torch.long))[i].to(self.device)

            out_i = self.model(z=z_i, pos=pos_i, batch=batch_i)

            preds_i, targets_i = {}, {}
            for task in self.tasks:
                if task in out_i and task in batch:
                    t = batch[task]
                    if isinstance(t, list):
                        t_i = t[i]
                    elif t.dim() >= 1:
                        t_i = t[i]
                    else:
                        t_i = t
                    t_i = t_i.to(self.device)
                    if norm_stats is not None:
                        t_i = norm_stats.normalize(task, t_i.unsqueeze(0)).squeeze(0)
                    targets_i[task] = t_i
                    preds_i[task] = out_i[task].squeeze(0)

            loss_i, per_task_i = self.criterion(preds_i, targets_i)
            total_loss = total_loss + loss_i
            for k, v in per_task_i.items():
                per_task[k] = per_task.get(k, 0.0) + v

        return total_loss / B, {k: v / B for k, v in per_task.items()}

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
                print("Epoch {:4d} | train loss: {:.4f} | val loss: {:.4f}".format(
                    epoch, train_m["loss"], val_m["loss"]))

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
