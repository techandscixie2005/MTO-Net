#!/usr/bin/env python3
"""Train MTO-Net for a specific stage (A/B/C)."""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mto.mto_model import MTONet
from src.mto.mto_readout import make_readout
from src.mto.training import Trainer, NormalizationStats
from src.mto.dataset_qm9s import QM9SDataset, collate_fn


def load_processed(data_dir, max_mols=0):
    """Load processed QM9S data."""
    train_path = os.path.join(data_dir, "train.pt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} not found. Run prepare_qm9s.py first.")

    datasets = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.pt")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if max_mols > 0 and split == "train":
            data = data[:max_mols]
        datasets[split] = data

    return datasets


def get_stage_tasks(stage):
    if stage == "stage_a":
        return ["mu", "alpha"]
    elif stage == "stage_b":
        return ["mu", "alpha", "ir", "raman"]
    elif stage == "stage_c":
        return ["mu", "alpha", "ir", "raman", "uv"]
    else:
        raise ValueError(f"Unknown stage: {stage}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--stage", default="stage_a", choices=["stage_a", "stage_b", "stage_c"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--maxl", type=int, default=3)
    parser.add_argument("--num-block", type=int, default=3)
    parser.add_argument("--rc", type=float, default=5.0)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage: {args.stage}")
    print(f"Data: {args.data_dir}")

    # Load data
    print("Loading data...")
    datasets = load_processed(args.data_dir)
    tasks = get_stage_tasks(args.stage)
    print(f"Tasks: {tasks}")
    print(f"Train: {len(datasets[train])} Val: {len(datasets[val])} Test: {len(datasets[test])}")

    # Convert to proper format for collate
    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_ds = ListDataset(datasets["train"])
    val_ds = ListDataset(datasets["val"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    # Compute normalization stats on training split
    norm_stats = NormalizationStats()
    print("Computing normalization stats...")
    all_train = datasets["train"]
    for task in tasks:
        tensors = []
        for sample in all_train:
            if task in sample and isinstance(sample[task], torch.Tensor):
                tensors.append(sample[task].reshape(-1))
        if tensors:
            norm_stats.fit_tensors(task, tensors)
            print(f"  {task}: mean={norm_stats.stats[task][mean]:.4f} std={norm_stats.stats[task][std]:.4f}")

    # Build model
    readout_tasks = {t: ({3: 3, 6: 6}.get(t, 3501 if t in ("ir", "raman") else 601)) for t in tasks}
    model = MTONet(
        feature_dim=args.feature_dim,
        tasks=readout_tasks,
        detanet_kwargs={"maxl": args.maxl, "num_block": args.num_block, "rc": args.rc},
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # Train
    ckpt_dir = os.path.join(args.checkpoint_dir, f"{args.stage}_seed{args.seed}")
    trainer = Trainer(model, device, tasks, lr=args.lr)
    history, best_path = trainer.fit(
        train_loader, val_loader, args.epochs,
        norm_stats=norm_stats, checkpoint_dir=ckpt_dir,
    )

    # Save normalization stats
    metrics_dir = "outputs/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    norm_stats.save(os.path.join(metrics_dir, f"{args.stage}_seed{args.seed}_norm.json"))

    # Save final metrics
    final_metrics = {
        "stage": args.stage,
        "seed": args.seed,
        "epochs": args.epochs,
        "params": total_params,
        "device": str(device),
        "best_val_loss": history["val"][-1]["loss"] if history["val"] else None,
        "final_train_loss": history["train"][-1]["loss"] if history["train"] else None,
        "history": history,
    }
    with open(os.path.join(metrics_dir, f"{args.stage}_seed{args.seed}_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)

    print(f"\nBest checkpoint: {best_path}")
    print(f"Final val loss: {final_metrics[best_val_loss]}")


if __name__ == "__main__":
    main()
