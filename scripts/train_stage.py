#!/usr/bin/env python3
"""Train MTO-Net for a specific stage (A/B/C)."""
import argparse, json, os, sys, time, subprocess
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mto.mto_model import MTONet
from src.mto.training import Trainer, NormalizationStats
from src.mto.dataset_qm9s import load_qm9s_raw, QM9SDataset, make_split, collate_batch

def get_stage_tasks(stage):
    tasks = {"stage_a": ["mu", "alpha"],
             "stage_b": ["mu", "alpha", "ir", "raman"],
             "stage_c": ["mu", "alpha", "ir", "raman", "uv"]}
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

    pt_path = os.path.join(args.data_dir, "qm9s.pt")
    if not os.path.exists(pt_path):
        pt_path_alt = os.path.join("data/qm9s", "qm9s.pt")
        if os.path.exists(pt_path_alt):
            pt_path = pt_path_alt
        else:
            raise FileNotFoundError("qm9s.pt not found")

    print("Loading QM9S from", pt_path)
    raw_data = load_qm9s_raw(pt_path)
    print("Loaded", len(raw_data), "molecules")

    dataset = QM9SDataset(raw_data)
    splits = make_split(dataset, train_frac=0.8, val_frac=0.1, seed=args.seed)
    print("Train:", len(splits["train"]), "Val:", len(splits["val"]), "Test:", len(splits["test"]))

    tasks = get_stage_tasks(args.stage)
    print("Tasks:", tasks)

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.data = [dataset[i] for i in subset.indices]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_ds = ListDataset(splits["train"])
    val_ds = ListDataset(splits["val"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    # Normalization stats on training split (sampled)
    norm_stats = NormalizationStats()
    print("Computing normalization stats...")
    for task in tasks:
        tensors = [s[task] for s in train_ds.data[:2000] if task in s]
        if tensors:
            norm_stats.fit_tensors(task, tensors)
            s = norm_stats.stats[task]
            print("  " + task + ": mean={:.4f} std={:.4f}".format(s["mean"], s["std"]))

    out_dims = {"mu": 3, "alpha": 9, "ir": 3501, "raman": 3501, "uv": 601}
    model = MTONet(feature_dim=args.feature_dim, tasks={t: out_dims[t] for t in tasks},
                   detanet_kwargs={"maxl": args.maxl, "num_block": args.num_block, "rc": args.rc})
    n_params = sum(p.numel() for p in model.parameters())
    print("Model params:", f"{n_params:,}")

    git_hash = "unknown"
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        if r.returncode == 0:
            git_hash = r.stdout.strip()
    except Exception:
        pass

    ckpt_dir = os.path.join(args.checkpoint_dir, args.stage + "_seed" + str(args.seed))
    trainer = Trainer(model, device, tasks, lr=args.lr)
    history, best_path = trainer.fit(train_loader, val_loader, args.epochs,
                                     norm_stats=norm_stats, checkpoint_dir=ckpt_dir)

    metrics_dir = "outputs/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    norm_stats.save(os.path.join(metrics_dir, args.stage + "_seed" + str(args.seed) + "_norm.json"))

    final_metrics = {
        "stage": args.stage, "seed": args.seed, "epochs": args.epochs,
        "params": n_params, "device": str(device),
        "best_val_loss": history["val"][-1]["loss"] if history["val"] else None,
        "final_train_loss": history["train"][-1]["loss"] if history["train"] else None,
        "git_hash": git_hash, "best_checkpoint": best_path, "tasks": tasks,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(metrics_dir, args.stage + "_seed" + str(args.seed) + "_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)

    print("\nBest checkpoint:", best_path)
    print("Final val loss:", final_metrics["best_val_loss"])

if __name__ == "__main__":
    main()
