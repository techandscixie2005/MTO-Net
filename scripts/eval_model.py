#!/usr/bin/env python3
"""Evaluate a trained MTO-Net checkpoint."""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mto.mto_model import MTONet
from src.mto.training import NormalizationStats, compute_metrics
from src.mto.dataset_qm9s import collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-metrics", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint epoch: {ckpt.get(epoch, unknown)}")
    print(f"Val loss at save: {ckpt.get(val_loss, unknown)}")

    # Load data
    data_path = os.path.join(args.data_dir, f"{args.split}.pt")
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found")
        sys.exit(1)
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    class ListDS(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            return self.data[i]

    loader = DataLoader(ListDS(data), batch_size=args.batch_size, collate_fn=collate_fn)

    # Detect tasks from checkpoint
    tasks = ckpt.get("tasks", ["mu", "alpha"])
    print(f"Tasks: {tasks}")

    # Build model
    readout_tasks = {t: 3 if t == "mu" else (6 if t == "alpha" else (3501 if t in ("ir", "raman") else 601)) for t in tasks}
    model = MTONet(tasks=readout_tasks).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load normalization
    norm_stats = None
    if args.norm_stats and os.path.exists(args.norm_stats):
        norm_stats = NormalizationStats.load(args.norm_stats)

    # Evaluate
    all_preds = {t: [] for t in tasks}
    all_targets = {t: [] for t in tasks}

    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            pos = batch["pos"].to(device)
            batch_idx = batch.get("batch", torch.zeros(len(z), dtype=torch.long)).to(device)
            preds = model(z=z, pos=pos, batch=batch_idx)
            for t in tasks:
                if t in preds and t in batch:
                    p = preds[t].cpu()
                    trg = batch[t].cpu()
                    if norm_stats:
                        p = norm_stats.denormalize(t, p)
                    all_preds[t].append(p)
                    all_targets[t].append(trg)

    for t in tasks:
        all_preds[t] = torch.cat(all_preds[t])
        all_targets[t] = torch.cat(all_targets[t])

    metrics = compute_metrics(all_preds, all_targets)
    print("\nMetrics:")
    for task, m in metrics.items():
        print(f"  {task}: MAE={m[mae]:.4f} RMSE={m[rmse]:.4f} R2={m[r2]:.4f}")

    if args.output_metrics:
        os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved: {args.output_metrics}")


if __name__ == "__main__":
    main()
