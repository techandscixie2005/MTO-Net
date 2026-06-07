#!/usr/bin/env python3
"""Train MTO-Net for a specific stage (A/B/C)."""
import argparse, json, os, sys, time, subprocess
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mto.mto_model import MTONet
from src.mto.training import Trainer, NormalizationStats
from src.mto.dataset_qm9s import load_qm9s_raw, QM9SDataset, make_split, collate_fn

def get_stage_tasks(stage):
    tasks = {
        "stage_a": ["mu", "alpha"],
        "stage_b": ["mu", "alpha", "ir", "raman"],
        "stage_c": ["mu", "alpha", "ir", "raman", "uv"],
    }
    return tasks[stage]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--stage", default="stage_a")
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
    print("Device:", str(device))
    print("Stage:", args.stage)
    print("Data:", args.data_dir)

    # Load raw QM9S data
    pt_path = os.path.join(args.data_dir, "qm9s.pt")
    if not os.path.exists(pt_path):
        pt_path_alt = os.path.join("data/qm9s", "qm9s.pt")
        if os.path.exists(pt_path_alt):
            pt_path = pt_path_alt
        else:
            raise FileNotFoundError("qm9s.pt not found at " + pt_path + " or " + pt_path_alt)

    print("Loading QM9S from", pt_path)
    raw_data = load_qm9s_raw(pt_path)
    print("Loaded", len(raw_data), "molecules")

    # Split into train/val/test
    dataset = QM9SDataset(raw_data)
    splits = make_split(dataset, train_frac=0.8, val_frac=0.1, seed=args.seed)
    print("Train:", len(splits["train"]), "Val:", len(splits["val"]), "Test:", len(splits["test"]))

    tasks = get_stage_tasks(args.stage)
    print("Tasks:", tasks)

    # Build loaders from dataset subsets
    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.data = []
            for idx in subset.indices:
                self.data.append(dataset[idx])
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_ds = ListDataset(splits["train"])
    val_ds = ListDataset(splits["val"])
    test_ds = ListDataset(splits["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Normalization stats on training split
    norm_stats = NormalizationStats()
    print("Computing normalization stats...")
    all_train = train_ds
    for task in tasks:
        tensors = []
        for sample in all_train:
            if task in sample and isinstance(sample[task], torch.Tensor):
                tensors.append(sample[task].reshape(-1))
        if tensors:
            norm_stats.fit_tensors(task, tensors)
            s = norm_stats.stats[task]
            print("  " + task + ": mean=" + "{:.4f}".format(s["mean"]) + " std=" + "{:.4f}".format(s["std"]))

    # Build model
    out_dims = {}
    for t in tasks:
        if t == "mu":
            out_dims[t] = 3
        elif t == "alpha":
            out_dims[t] = 9  # 3x3 polar flatten
        elif t in ("ir", "raman"):
            out_dims[t] = 3501
        elif t == "uv":
            out_dims[t] = 601
        else:
            out_dims[t] = 3

    model = MTONet(feature_dim=args.feature_dim, tasks=out_dims,
                   detanet_kwargs={"maxl": args.maxl, "num_block": args.num_block, "rc": args.rc})
    n_params = sum(p.numel() for p in model.parameters())
    print("Model params:", f"{n_params:,}")

    # Get git hash
    git_hash = "unknown"
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=".")
        if r.returncode == 0:
            git_hash = r.stdout.strip()
    except Exception:
        pass

    # Train
    ckpt_dir = os.path.join(args.checkpoint_dir, args.stage + "_seed" + str(args.seed))
    trainer = Trainer(model, device, tasks, lr=args.lr)
    history, best_path = trainer.fit(train_loader, val_loader, args.epochs,
                                     norm_stats=norm_stats, checkpoint_dir=ckpt_dir)

    # Save normalization stats
    metrics_dir = "outputs/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    norm_path = os.path.join(metrics_dir, args.stage + "_seed" + str(args.seed) + "_norm.json")
    norm_stats.save(norm_path)

    # Save metrics JSON
    final_metrics = {
        "stage": args.stage,
        "seed": args.seed,
        "epochs": args.epochs,
        "params": n_params,
        "device": str(device),
        "best_val_loss": history["val"][-1]["loss"] if history["val"] else None,
        "final_train_loss": history["train"][-1]["loss"] if history["train"] else None,
        "git_hash": git_hash,
        "best_checkpoint": best_path,
        "tasks": tasks,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metrics_path = os.path.join(metrics_dir, args.stage + "_seed" + str(args.seed) + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)

    print("\nBest checkpoint:", best_path)
    print("Final val loss:", final_metrics["best_val_loss"])

if __name__ == "__main__":
    main()
