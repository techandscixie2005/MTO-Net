#!/usr/bin/env python3
"""Run stage transfer stability experiment (intra-Stage-A).

Since Stage B/C labels are direct spectrum CSVs (no physical Hessian/normal modes),
we run intra-Stage-A transfer to test MTO subspace transfer:

1. mu-only training → evaluate alpha (does MTO learned for mu transfer to alpha?)
2. alpha-only training → evaluate mu (does MTO learned for alpha transfer to mu?)
3. mu+alpha training (both tasks, baseline)

MTO subspace similarity, slot activity correlation, and prediction transfer
metrics are computed across conditions.
"""
import argparse, json, os, sys, time
import torch
import torch.nn as nn
import numpy as np

_proj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "third_party", "DetaNet"))

from src.mto.compat import *  # noqa
from src.mto.mto_model import MTONet
from src.mto.dataset_qm9s import load_qm9s_raw, collate_batch
from src.mto.data_splits import generate_split, split_indices_for_seed
from src.mto.stability import extract_mto_contributions, stage_stability


class LazySubset(torch.utils.data.Dataset):
    def __init__(self, data_list, indices):
        self.data = data_list; self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.data[self.indices[idx]]


class SimpleTrainer:
    def __init__(self, model, device, tasks, lr=1e-3):
        self.model = model
        self.device = device
        self.tasks = tasks
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.mu_mean = 0.0; self.mu_std = 1.0
        self.alpha_mean = 0.0; self.alpha_std = 1.0

    def set_norm(self, mu_mean, mu_std, alpha_mean, alpha_std):
        self.mu_mean = mu_mean; self.mu_std = mu_std
        self.alpha_mean = alpha_mean; self.alpha_std = alpha_std

    def _normalize(self, task, t):
        if task == "mu": return (t - self.mu_mean) / max(self.mu_std, 1e-6)
        if task == "alpha": return (t - self.alpha_mean) / max(self.alpha_std, 1e-6)
        return t

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0; n = 0
        for batch in loader:
            z = batch["z"].to(self.device)
            pos = batch["pos"].to(self.device)
            batch_idx = batch["batch"].to(self.device)
            out = self.model(z=z, pos=pos, batch=batch_idx)
            loss = 0.0
            for task in self.tasks:
                if task in out and task in batch:
                    pred = out[task]
                    target = self._normalize(task, batch[task].to(self.device))
                    loss += ((pred - target) ** 2).mean()
            if loss > 0:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0; n = 0
        for batch in loader:
            z = batch["z"].to(self.device)
            pos = batch["pos"].to(self.device)
            batch_idx = batch["batch"].to(self.device)
            out = self.model(z=z, pos=pos, batch=batch_idx)
            for task in self.tasks:
                if task in out and task in batch:
                    pred = out[task]
                    target = self._normalize(task, batch[task].to(self.device))
                    total_loss += ((pred - target) ** 2).mean().item()
                    n += 1
        return total_loss / max(n, 1)

    def fit(self, train_loader, val_loader, epochs, ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        best_val = float("inf"); best_path = None
        history = {"train": [], "val": []}
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save({"model_state_dict": self.model.state_dict(),
                            "epoch": epoch, "val_loss": val_loss}, best_path)
            if epoch % 5 == 0:
                print(f"  epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")
        return best_val, best_path or os.path.join(ckpt_dir, "best.pt"), history


@torch.no_grad()
def evaluate_model(model, loader, device, task_list, mu_mean=0.0, mu_std=1.0, alpha_mean=0.0, alpha_std=1.0):
    """Evaluate model returning both raw and normalized MAE."""
    model.eval()
    raw_errors = {t: [] for t in task_list}
    norm_errors = {t: [] for t in task_list}
    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        batch_idx = batch["batch"].to(device)
        out = model(z=z, pos=pos, batch=batch_idx)
        for t in task_list:
            if t in out and t in batch:
                target = batch[t].to(device)
                raw_diff = (out[t] - target).abs().mean().item()
                raw_errors[t].append(raw_diff)
                if t == "mu":
                    target_n = (target - mu_mean) / max(mu_std, 1e-6)
                    pred_n = (out[t] - mu_mean) / max(mu_std, 1e-6)
                elif t == "alpha":
                    target_n = (target - alpha_mean) / max(alpha_std, 1e-6)
                    pred_n = (out[t] - alpha_mean) / max(alpha_std, 1e-6)
                else:
                    target_n = target; pred_n = out[t]
                norm_diff = (pred_n - target_n).abs().mean().item()
                norm_errors[t].append(norm_diff)
    raw = {t: float(np.mean(vs)) if vs else float("nan") for t, vs in raw_errors.items()}
    norm = {t: float(np.mean(vs)) if vs else float("nan") for t, vs in norm_errors.items()}
    return raw, norm


@torch.no_grad()
def compute_mto_subspace(model, loader, device, top_r=5):
    """Extract MTO coefficient subspace from the MTO module output."""
    coeffs_list = []
    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        batch_idx = batch["batch"].to(device)
        out = model(z=z, pos=pos, batch=batch_idx, return_mto=True)
        if "coeff" in out and out["coeff"] is not None:
            coeffs_list.append(out["coeff"].detach().cpu())
    if not coeffs_list:
        return None
    # coeff shape is (B, K_max, N_atoms). Average over atoms -> (B, K_max), pad K.
    max_K = max(c.shape[1] for c in coeffs_list)
    padded = []
    for c in coeffs_list:
        c_m = c.mean(dim=-1)  # average over atoms -> (B, K_max)
        if c_m.shape[1] < max_K:
            c_m = torch.nn.functional.pad(c_m, (0, max_K - c_m.shape[1]))
        padded.append(c_m)
    all_coeffs = torch.cat(padded, dim=0)
    U, S, V = torch.svd(all_coeffs.float())
    return U[:, :top_r]


def subspace_similarity(U1, U2):
    """Cosine similarity of subspaces via SVD of U1^T U2."""
    if U1 is None or U2 is None:
        return None
    M = U1.T @ U2
    _, s, _ = torch.svd(M)
    return float(s.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/qm9s/subset_medium")
    parser.add_argument("--output-dir", default="outputs/stage_transfer")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--maxl", type=int, default=3)
    parser.add_argument("--num-block", type=int, default=3)
    parser.add_argument("--rc", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pt_path = os.path.join(args.data_dir, "qm9s.pt") if os.path.isdir(args.data_dir) else args.data_dir
    data = load_qm9s_raw(pt_path)
    print(f"Molecules: {len(data)}")

    conditions = {
        "mu_only": {"mu": 3},
        "alpha_only": {"alpha": 9},
        "mu_alpha": {"mu": 3, "alpha": 9},
    }
    eval_tasks = ["mu", "alpha"]

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*60}\n  STAGE TRANSFER — seed={seed}\n{'='*60}")

        split = generate_split(len(data), train_frac=0.7, val_frac=0.15, seed=seed)
        train_idx, val_idx, test_idx = split_indices_for_seed(split, seed)
        train_sub = LazySubset(data, train_idx)
        val_sub = LazySubset(data, val_idx)
        test_sub = LazySubset(data, test_idx)

        mu_vals, alpha_vals = [], []
        for i in range(min(200, len(train_idx))):
            mol = train_sub[i]
            mu_vals.append(mol.dipole.clone())
            alpha_vals.append(mol.polar.reshape(-1).clone())
        mu_all = torch.cat([t.reshape(-1) for t in mu_vals])
        alpha_all = torch.cat([t.reshape(-1) for t in alpha_vals])
        mu_mean, mu_std = float(mu_all.mean()), max(float(mu_all.std()), 1e-6)
        alpha_mean, alpha_std = float(alpha_all.mean()), max(float(alpha_all.std()), 1e-6)

        train_loader = torch.utils.data.DataLoader(
            train_sub, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = torch.utils.data.DataLoader(
            val_sub, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        test_loader = torch.utils.data.DataLoader(
            test_sub, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        dk = {"maxl": args.maxl, "num_block": args.num_block, "rc": args.rc, "device": device}

        seed_results = {"seed": seed, "conditions": {}}
        models_by_cond = {}

        for cond_name, cond_tasks in conditions.items():
            print(f"\n  --- Condition: {cond_name} ---")
            print(f"  Train tasks: {list(cond_tasks.keys())}")

            model = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                          readout_hidden_dim=64, tasks=cond_tasks, detanet_kwargs=dk,
                          use_activity_gate=True, activity_mode="simple").to(device)

            trainer = SimpleTrainer(model, device, cond_tasks, lr=args.lr)
            trainer.set_norm(mu_mean, mu_std, alpha_mean, alpha_std)
            best_val, ckpt_path, hist = trainer.fit(
                train_loader, val_loader, args.epochs,
                os.path.join(args.output_dir, f"seed{seed}", cond_name))

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            models_by_cond[cond_name] = model

            # Evaluate on BOTH tasks (transfer)
            test_raw, test_norm = evaluate_model(model, test_loader, device, eval_tasks,
                                                 mu_mean, mu_std, alpha_mean, alpha_std)
            seed_results["conditions"][cond_name] = {
                "train_tasks": list(cond_tasks.keys()),
                "best_val": best_val,
                "test_raw": test_raw,
                "test_norm": test_norm,
                "history": hist,
            }
            print(f"  Test raw: {test_raw} | norm: {test_norm}")

        # Compute MTO subspace similarity between conditions
        print("\n  --- MTO Subspace Similarity ---")
        for i, c1 in enumerate(conditions):
            for c2 in list(conditions.keys())[i:]:
                U1 = compute_mto_subspace(models_by_cond.get(c1), test_loader, device)
                U2 = compute_mto_subspace(models_by_cond.get(c2), test_loader, device)
                sim = subspace_similarity(U1, U2)
                key = f"{c1}_vs_{c2}"
                seed_results.setdefault("subspace_sim", {})[key] = sim
                print(f"  {c1} vs {c2}: S_sub = {sim:.4f}" if sim else f"  {c1} vs {c2}: N/A")

        all_results.append(seed_results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "stage_transfer_results.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print("\n" + "=" * 90)
    print("  STAGE TRANSFER SUMMARY")
    print("=" * 90)
    print(f"{'Seed':<6s} {'Condition':<14s} {'mu_raw':>8s} {'mu_norm':>8s} {'a_raw':>8s} {'a_norm':>8s} {'S_sub':>12s}")
    print("-" * 90)
    for sr in all_results:
        s = sr["seed"]
        for cn in conditions:
            tr = sr["conditions"].get(cn, {}).get("test_raw", {})
            tn = sr["conditions"].get(cn, {}).get("test_norm", {})
            sub = sr.get("subspace_sim", {}).get("mu_only_vs_alpha_only")
            sub_str = f"{sub:.4f}" if sub is not None else "N/A"
            print(f"{s:<6d} {cn:<14s} {tr.get('mu', 0):>8.4f} {tn.get('mu', 0):>8.4f} "
                  f"{tr.get('alpha', 0):>8.4f} {tn.get('alpha', 0):>8.4f} {sub_str:>12s}")

    print(f"\nResults saved to {metrics_path}")


if __name__ == "__main__":
    main()
