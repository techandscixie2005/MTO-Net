#!/usr/bin/env python3
"""Run baseline comparisons at smoke scale on QM9S subset.

Baselines compared:
1. Full MTO-Net (signed, valence-adaptive)
2. No-sign MTO (softmax-only routing)
3. Fixed-K MTO (K=20 fixed, using baseline class)
4. Direct readout (no MTO, sum pool)
5. Attention pooling (learned attention weights)

Saves metrics for comparison figure generation.
"""
import argparse, json, os, sys, time
import torch
import torch.nn as nn
import numpy as np

# proj_root = scripts/eval/ -> scripts/ -> repo root
_proj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "third_party", "DetaNet"))

from src.mto.compat import *  # noqa: must be before any DetaNet imports
from src.mto.mto_model import MTONet
from src.mto.dataset_qm9s import load_qm9s_raw, collate_batch
from src.mto.data_splits import generate_split, split_indices_for_seed
from src.mto.losses import CompositeLoss


class LazySubset(torch.utils.data.Dataset):
    def __init__(self, data_list, indices):
        self.data = data_list; self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.data[self.indices[idx]]


class SimpleTrainer:
    """Minimal trainer that works with any model taking (z, pos, batch=None)."""
    def __init__(self, model, device, tasks, lr=1e-3):
        self.model = model
        self.device = device
        self.tasks = tasks
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        # Simple identity normalization (raw MAE)
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
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)
            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(ckpt_dir, "best.pt")
                torch.save({"model_state_dict": self.model.state_dict()}, best_path)
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")
        return best_val, best_path or os.path.join(ckpt_dir, "best.pt")


def evaluate_model(model, loader, device, tasks, mu_mean=0.0, mu_std=1.0, alpha_mean=0.0, alpha_std=1.0):
    """Evaluate model returning both raw and normalized MAE."""
    model.eval()
    raw_errors = {t: [] for t in tasks}
    norm_errors = {t: [] for t in tasks}
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            pos = batch["pos"].to(device)
            batch_idx = batch["batch"].to(device)
            out = model(z=z, pos=pos, batch=batch_idx)
            for t in tasks:
                if t in out and t in batch:
                    target = batch[t].to(device)
                    raw_diff = (out[t] - target).abs().mean().item()
                    raw_errors[t].append(raw_diff)
                    if t == "mu":
                        target_norm = (target - mu_mean) / max(mu_std, 1e-6)
                        pred_norm = (out[t] - mu_mean) / max(mu_std, 1e-6)
                    elif t == "alpha":
                        target_norm = (target - alpha_mean) / max(alpha_std, 1e-6)
                        pred_norm = (out[t] - alpha_mean) / max(alpha_std, 1e-6)
                    else:
                        target_norm = target
                        pred_norm = out[t]
                    norm_diff = (pred_norm - target_norm).abs().mean().item()
                    norm_errors[t].append(norm_diff)
    raw = {t: float(np.mean(vs)) if vs else float("nan") for t, vs in raw_errors.items()}
    norm = {t: float(np.mean(vs)) if vs else float("nan") for t, vs in norm_errors.items()}
    return raw, norm


def run_baseline(name, model, tasks, train_loader, val_loader, test_loader,
                 mu_mean, mu_std, alpha_mean, alpha_std, device, args):
    print(f"\n{'='*60}\n  Baseline: {name}\n{'='*60}")
    ckpt_dir = os.path.join(args.checkpoint_dir, name.replace(" ", "_"))
    trainer = SimpleTrainer(model, device, tasks, lr=args.lr)
    trainer.set_norm(mu_mean, mu_std, alpha_mean, alpha_std)
    _, best_path = trainer.fit(train_loader, val_loader, args.epochs, ckpt_dir)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    raw_val, norm_val = evaluate_model(model, val_loader, device, tasks,
                                       mu_mean, mu_std, alpha_mean, alpha_std)
    raw_test, norm_test = evaluate_model(model, test_loader, device, tasks,
                                         mu_mean, mu_std, alpha_mean, alpha_std)
    print(f"  Val raw:  {raw_val} | Val norm: {norm_val}")
    print(f"  Test raw: {raw_test} | Test norm: {norm_test}")
    return {"name": name, "val_raw": raw_val, "val_norm": norm_val,
            "test_raw": raw_test, "test_norm": norm_test,
            "checkpoint": best_path,
            "params": sum(p.numel() for p in model.parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/qm9s/subset_smoke")
    parser.add_argument("--checkpoint-dir", default="outputs/baselines")
    parser.add_argument("--metrics-dir", default="outputs/metrics/baselines")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--maxl", type=int, default=1)
    parser.add_argument("--num-block", type=int, default=1)
    parser.add_argument("--rc", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = {"mu": 3, "alpha": 9}
    print(f"Device: {device}")

    # Load data
    pt_path = os.path.join(args.data_dir, "qm9s.pt")
    data = load_qm9s_raw(pt_path)
    n = len(data)
    print(f"Molecules: {n}")

    split = generate_split(int(n), train_frac=0.7, val_frac=0.15, seed=0)
    train_idx, val_idx, test_idx = split_indices_for_seed(split, 0)
    train_subset = LazySubset(data, train_idx)
    val_subset = LazySubset(data, val_idx)
    test_subset = LazySubset(data, test_idx)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Compute normalization stats from train set
    mu_vals, alpha_vals = [], []
    for i in range(min(200, len(train_idx))):
        mol = train_subset[i]
        mu_vals.append(mol.dipole.clone())
        alpha_vals.append(mol.polar.reshape(-1).clone())
    mu_all = torch.cat([t.reshape(-1) for t in mu_vals])
    alpha_all = torch.cat([t.reshape(-1) for t in alpha_vals])
    mu_mean, mu_std = float(mu_all.mean()), max(float(mu_all.std()), 1e-6)
    alpha_mean, alpha_std = float(alpha_all.mean()), max(float(alpha_all.std()), 1e-6)
    print(f"Norm: mu({mu_mean:.3f}+/-{mu_std:.3f}) alpha({alpha_mean:.3f}+/-{alpha_std:.3f})")

    # Build data loaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    dk = {"maxl": args.maxl, "num_block": args.num_block, "rc": args.rc, "device": device}
    results = []

    # 1. Full MTO-Net
    print("\n[1/5] Full MTO-Net")
    m_full = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                    readout_hidden_dim=64, tasks=tasks, detanet_kwargs=dk,
                    use_activity_gate=True, activity_mode="simple").to(device)
    results.append(run_baseline("full_mto", m_full, tasks, train_loader, val_loader,
                                test_loader, mu_mean, mu_std, alpha_mean, alpha_std, device, args))

    # 2. No-sign MTO (zero out sign MLPs)
    print("\n[2/5] No-sign MTO")
    m_nosign = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                      readout_hidden_dim=64, tasks=tasks, detanet_kwargs=dk,
                      use_activity_gate=True, activity_mode="simple")
    for m in [m_nosign.mto.sign_mlp_l0, m_nosign.mto.sign_mlp_l1, m_nosign.mto.sign_mlp_l2]:
        for layer in m:
            if hasattr(layer, 'weight'):
                nn.init.constant_(layer.weight, 0)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    m_nosign = m_nosign.to(device)
    results.append(run_baseline("no_sign_mto", m_nosign, tasks, train_loader, val_loader,
                                test_loader, mu_mean, mu_std, alpha_mean, alpha_std, device, args))

    # 3. Fixed-K MTO
    print("\n[3/5] Fixed-K MTO")
    from src.mto.baselines import FixedKTokenBaseline
    m_fixedk = FixedKTokenBaseline(m_full.backbone, tasks, K_fixed=20,
                                   feature_dim=args.feature_dim, hidden_dim=32).to(device)
    results.append(run_baseline("fixed_k_mto", m_fixedk, tasks, train_loader, val_loader,
                                test_loader, mu_mean, mu_std, alpha_mean, alpha_std, device, args))

    # 4. Direct readout
    print("\n[4/5] Direct readout")
    from src.mto.baselines import DirectReadoutBaseline
    m_direct = DirectReadoutBaseline(m_full.backbone, tasks, hidden_dim=64).to(device)
    results.append(run_baseline("direct_readout", m_direct, tasks, train_loader, val_loader,
                                test_loader, mu_mean, mu_std, alpha_mean, alpha_std, device, args))

    # 5. Attention pooling
    print("\n[5/5] Attention pooling")
    from src.mto.baselines import AttentionPoolingBaseline
    m_attn = AttentionPoolingBaseline(m_full.backbone, tasks, hidden_dim=64).to(device)
    results.append(run_baseline("attention_pooling", m_attn, tasks, train_loader, val_loader,
                                test_loader, mu_mean, mu_std, alpha_mean, alpha_std, device, args))

    # Save results
    os.makedirs(args.metrics_dir, exist_ok=True)
    metrics_path = os.path.join(args.metrics_dir, "baseline_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {metrics_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON SUMMARY (mu/alpha MAE)")
    print("=" * 70)
    print(f"{'Method':<25s} {'mu_raw':>8s} {'mu_norm':>8s} {'a_raw':>8s} {'a_norm':>8s} {'Params':>10s}")
    print("-" * 67)
    for r in results:
        print(f"{r['name']:<25s} {r['test_raw'].get('mu', 0):>8.4f} {r['test_norm'].get('mu', 0):>8.4f} "
              f"{r['test_raw'].get('alpha', 0):>8.4f} {r['test_norm'].get('alpha', 0):>8.4f} {r['params']:>10d}")
    print("=" * 70)


if __name__ == "__main__":
    main()
