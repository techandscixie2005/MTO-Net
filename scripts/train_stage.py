#!/usr/bin/env python3
"""Train MTO-Net for a specific stage (A/B/C)."""
import argparse, json, os, sys, time, subprocess, datetime
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mto.mto_model import MTONet
from src.mto.training import Trainer, NormalizationStats
from src.mto.dataset_qm9s import load_qm9s_raw, collate_batch


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)


def get_stage_tasks(stage):
    return {"stage_a": ["mu", "alpha"],
            "stage_b": ["mu", "alpha", "ir", "raman"],
            "stage_c": ["mu", "alpha", "ir", "raman", "uv"]}[stage]


class LazySubset(torch.utils.data.Dataset):
    def __init__(self, data_list, indices):
        self.data = data_list
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.data[self.indices[idx]]


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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("Device:", str(device))
    log("Stage:", args.stage)

    pt_path = os.path.join(args.data_dir, "qm9s.pt")
    if not os.path.exists(pt_path):
        pt_path = os.path.join("data/qm9s", "qm9s.pt")
    log("Loading QM9S from", pt_path)
    raw_data = load_qm9s_raw(pt_path)
    log("Loaded", len(raw_data), "molecules")

    n_total = len(raw_data)
    indices = list(range(n_total))
    import random
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    log("Train:", n_train, "Val:", n_val, "Test:", n_total - n_train - n_val)

    tasks = get_stage_tasks(args.stage)
    log("Tasks:", tasks)

    train_loader = DataLoader(LazySubset(raw_data, train_indices),
                              batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_batch)
    val_loader = DataLoader(LazySubset(raw_data, val_indices),
                            batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_batch)

    norm_stats = NormalizationStats()
    log("Computing normalization stats (sampling 2000)...")
    sample_indices = indices[:2000]
    for task in tasks:
        tensors = []
        for i in sample_indices:
            mol = raw_data[i]
            if task == "mu" and hasattr(mol, "dipole"):
                tensors.append(mol.dipole.float().reshape(-1))
            elif task == "alpha" and hasattr(mol, "polar"):
                tensors.append(mol.polar.float().reshape(-1))
        if tensors:
            norm_stats.fit_tensors(task, tensors)
            s = norm_stats.stats[task]
            log("  " + task + ": mean={:.4f} std={:.4f}".format(s["mean"], s["std"]))

    out_dims = {"mu": 3, "alpha": 9, "ir": 3501, "raman": 3501, "uv": 601}
    model = MTONet(feature_dim=args.feature_dim, tasks={t: out_dims[t] for t in tasks},
                   detanet_kwargs={"maxl": args.maxl, "num_block": args.num_block, "rc": args.rc})
    n_params = sum(p.numel() for p in model.parameters())
    log("Model params:", f"{n_params:,}")

    git_hash = "unknown"
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        if r.returncode == 0:
            git_hash = r.stdout.strip()
    except Exception:
        pass

    ckpt_dir = os.path.join(args.checkpoint_dir, args.stage + "_seed" + str(args.seed))
    trainer = Trainer(model, device, tasks, lr=args.lr)
    log("Starting training...")

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
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(metrics_dir, args.stage + "_seed" + str(args.seed) + "_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)

    log("Best checkpoint:", best_path)
    log("Final val loss:", final_metrics["best_val_loss"])


if __name__ == "__main__":
    main()
