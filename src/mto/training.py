"""Training utilities for MTO-Net."""
import json, os, random
import torch, torch.nn as nn

class NormalizationStats:
    def __init__(self):
        self.stats = {}
    def fit_tensors(self, key, tensors, max_samples=2000):
        if len(tensors) > max_samples:
            tensors = random.sample(tensors, max_samples)
        all_vals = torch.cat([t.reshape(-1) for t in tensors])
        s = float(all_vals.std().item())
        self.stats[key] = {"mean": float(all_vals.mean().item()),
                           "std": s if s > 1e-8 else 1.0}
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
        from .losses import CompositeLoss, default_task_weights
        self.model = model.to(device)
        self.device = device
        self.tasks = tasks
        self.criterion = CompositeLoss(task_weights or default_task_weights(tasks))
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10)

    def train_epoch(self, loader, norm_stats=None):
        self.model.train()
        total_loss, n, task_losses = 0.0, 0, {}
        for batch in loader:
            self.optimizer.zero_grad()
            loss, per_task = self._forward(batch, norm_stats)
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
            loss, per_task = self._forward(batch, norm_stats)
            total_loss += loss.item()
            n += 1
            for k, v in per_task.items():
                task_losses[k] = task_losses.get(k, 0.0) + v
        return {"loss": total_loss / n, "task_losses": {k: v / n for k, v in task_losses.items()}}

    def _forward(self, batch, norm_stats):
        z = batch["z"].to(self.device)
        pos = batch["pos"].to(self.device)
        batch_idx = batch["batch"].to(self.device)

        out = self.model(z=z, pos=pos, batch=batch_idx)

        targets = {}
        for task in self.tasks:
            if task in batch:
                t = batch[task]
                if isinstance(t, list):
                    t = torch.stack(t)
                targets[task] = t.to(self.device)
                if norm_stats is not None:
                    targets[task] = norm_stats.normalize(task, targets[task])

        preds = out
        return self.criterion(preds, targets)

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
                torch.save({"model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch, "val_loss": val_m["loss"], "history": history,
                            "train_loss": train_m["loss"]},
                           os.path.join(checkpoint_dir, "last.pt"))
            if val_m["loss"] < best_val_loss and checkpoint_dir:
                best_val_loss = val_m["loss"]
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save({"model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "epoch": epoch, "best_epoch": epoch,
                            "best_val_loss": best_val_loss, "val_metrics": val_m,
                            "train_loss": train_m["loss"], "history": history}, best_path)
            if epoch % log_interval == 0:
                print("Epoch {:4d} | train: {:.4f} | val: {:.4f}".format(
                    epoch, train_m["loss"], val_m["loss"]))
        return history, best_path

def compute_metrics(preds, targets):
    import numpy as np
    metrics = {}
    for task in preds:
        p = preds[task].detach().cpu().numpy()
        t = targets[task].detach().cpu().numpy()
        metrics[task] = {"mae": float(np.mean(np.abs(p - t))),
                         "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
                         "r2": float(1 - np.sum((t-p)**2)/(np.sum((t-np.mean(t))**2)+1e-10))}
    return metrics
