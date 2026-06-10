#!/usr/bin/env python3
"""Run frozen probe experiment: test MTO representation reusability.

Three conditions:
1. Frozen Stage A MTO encoder -> train new readout head
2. Frozen DetaNet backbone (no MTO) -> train new readout head
3. Random/fresh MTO encoder -> train readout (from-scratch baseline)

Each trained on mu+alpha or single tasks, compared by test MSE.
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


class LazySubset(torch.utils.data.Dataset):
    def __init__(self, data_list, indices):
        self.data = data_list; self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.data[self.indices[idx]]


def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_backbone_and_mto(model):
    """Freeze DetaNet backbone and MTO module. Only readout stays trainable."""
    for name, param in model.named_parameters():
        if "readout" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def reset_readout_heads(model, tasks, device, hidden_dim=64):
    """Replace readout with fresh randomly initialized heads."""
    from src.mto.mto_readout import MultiHeadReadout
    model.readout = MultiHeadReadout(
        feature_dim=model.feature_dim,
        hidden_dim=hidden_dim,
        tasks=tasks,
    )
    model.readout = model.readout.to(device)
    return model


class SimpleTrainer:
    def __init__(self, model, device, tasks, lr=1e-3):
        self.model = model
        self.device = device
        self.tasks = tasks
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-5)
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
def evaluate_model(model, loader, device, tasks):
    model.eval()
    errors = {t: [] for t in tasks}
    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        batch_idx = batch["batch"].to(device)
        out = model(z=z, pos=pos, batch=batch_idx)
        for t in tasks:
            if t in out and t in batch:
                diff = (out[t] - batch[t].to(device)).abs().mean().item()
                errors[t].append(diff)
    return {t: float(np.mean(vs)) if vs else float("nan")
            for t, vs in errors.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/qm9s/subset_medium")
    parser.add_argument("--stage-a-ckpt", default="outputs/checkpoints/stage_a_seed1/best.pt")
    parser.add_argument("--output-dir", default="outputs/frozen_probe")
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
    tasks = {"mu": 3, "alpha": 9}
    print(f"Device: {device}")

    # Load data
    pt_path = os.path.join(args.data_dir, "qm9s.pt") if os.path.isdir(args.data_dir) else args.data_dir
    data = load_qm9s_raw(pt_path)
    print(f"Molecules: {len(data)}")

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*60}\n  FROZEN PROBE — seed={seed}\n{'='*60}")

        split = generate_split(len(data), train_frac=0.7, val_frac=0.15, seed=seed)
        train_idx, val_idx, test_idx = split_indices_for_seed(split, seed)
        train_sub = LazySubset(data, train_idx)
        val_sub = LazySubset(data, val_idx)
        test_sub = LazySubset(data, test_idx)
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # Normalization stats
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

        seed_results = {"seed": seed}

        # -------------------------------------------------------
        # Condition 1: FROZEN Stage A MTO + fresh readout
        # -------------------------------------------------------
        print("\n[1/3] Frozen MTO encoder + fresh readout")
        ckpt = torch.load(args.stage_a_ckpt, map_location=device, weights_only=False)
        # Match Stage A checkpoint: feature_dim=128, mto_hidden_dim=64, readout_hidden_dim=128
        m_frozen = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=64,
                          readout_hidden_dim=128, tasks=tasks, detanet_kwargs=dk,
                          use_activity_gate=True, activity_mode="simple")

        # Handle missing/unexpected keys gracefully
        model_state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = m_frozen.load_state_dict(model_state, strict=False)
        if missing:
            print(f"  Note: missing keys: {len(missing)}")
        if unexpected:
            print(f"  Note: unexpected keys: {len(unexpected)}")

        m_frozen = m_frozen.to(device)
        freeze_backbone_and_mto(m_frozen)
        reset_readout_heads(m_frozen, tasks, device)
        n_frozen_trainable = count_params(m_frozen, trainable_only=True)
        print(f"  Trainable params (frozen backbone+MTO): {n_frozen_trainable}")

        trainer = SimpleTrainer(m_frozen, device, tasks, lr=args.lr)
        trainer.set_norm(mu_mean, mu_std, alpha_mean, alpha_std)
        best_val, ckpt_path, hist = trainer.fit(
            train_loader, val_loader, args.epochs,
            os.path.join(args.output_dir, f"seed{seed}", "frozen_mto"))

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m_frozen.load_state_dict(ckpt["model_state_dict"], strict=False)
        test_metrics = evaluate_model(m_frozen, test_loader, device, tasks)
        seed_results["frozen_mto"] = {
            "trainable_params": n_frozen_trainable,
            "best_val": best_val,
            "test": test_metrics,
            "history": hist,
        }
        print(f"  Frozen MTO test: {test_metrics}")

        # -------------------------------------------------------
        # Condition 2: FROZEN DetaNet (no MTO) + fresh readout
        # -------------------------------------------------------
        print("\n[2/3] Frozen DetaNet backbone + fresh readout")
        from src.mto.baselines import DirectReadoutBaseline

        # Build fresh model, freeze backbone
        m_direct = DirectReadoutBaseline(
            m_frozen.backbone if hasattr(m_frozen, 'backbone') else None,
            tasks, hidden_dim=64
        ).to(device)

        # Freeze backbone
        for name, param in m_direct.named_parameters():
            if "readout" not in name:
                param.requires_grad = False
        n_direct_trainable = count_params(m_direct, trainable_only=True)
        print(f"  Trainable params (frozen backbone only): {n_direct_trainable}")

        trainer2 = SimpleTrainer(m_direct, device, tasks, lr=args.lr)
        trainer2.set_norm(mu_mean, mu_std, alpha_mean, alpha_std)
        best_val2, ckpt_path2, hist2 = trainer2.fit(
            train_loader, val_loader, args.epochs,
            os.path.join(args.output_dir, f"seed{seed}", "frozen_direct"))

        ckpt2 = torch.load(ckpt_path2, map_location=device, weights_only=False)
        m_direct.load_state_dict(ckpt2.get("model_state_dict", ckpt2), strict=False)
        test_metrics2 = evaluate_model(m_direct, test_loader, device, tasks)
        seed_results["frozen_direct"] = {
            "trainable_params": n_direct_trainable,
            "best_val": best_val2,
            "test": test_metrics2,
            "history": hist2,
        }
        print(f"  Frozen DetaNet test: {test_metrics2}")

        # -------------------------------------------------------
        # Condition 3: FROM-SCRATCH MTO (trainable backbone+MTO)
        # -------------------------------------------------------
        print("\n[3/3] From-scratch MTO (full training, baseline)")
        m_scratch = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=64,
                           readout_hidden_dim=128, tasks=tasks, detanet_kwargs=dk,
                           use_activity_gate=True, activity_mode="simple").to(device)
        n_scratch = count_params(m_scratch, trainable_only=True)
        print(f"  Trainable params (full model): {n_scratch}")

        trainer3 = SimpleTrainer(m_scratch, device, tasks, lr=args.lr)
        trainer3.set_norm(mu_mean, mu_std, alpha_mean, alpha_std)
        best_val3, ckpt_path3, hist3 = trainer3.fit(
            train_loader, val_loader, args.epochs,
            os.path.join(args.output_dir, f"seed{seed}", "from_scratch"))

        ckpt3 = torch.load(ckpt_path3, map_location=device, weights_only=False)
        m_scratch.load_state_dict(ckpt3.get("model_state_dict", ckpt3), strict=False)
        test_metrics3 = evaluate_model(m_scratch, test_loader, device, tasks)
        seed_results["from_scratch"] = {
            "trainable_params": n_scratch,
            "best_val": best_val3,
            "test": test_metrics3,
            "history": hist3,
        }
        print(f"  From-scratch test: {test_metrics3}")

        all_results.append(seed_results)

    # Save all results
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "frozen_probe_results.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("  FROZEN PROBE SUMMARY")
    print("=" * 70)
    for sr in all_results:
        print(f"\nSeed {sr['seed']}:")
        for cond in ["frozen_mto", "frozen_direct", "from_scratch"]:
            if cond in sr:
                t = sr[cond]["test"]
                n = sr[cond]["trainable_params"]
                print(f"  {cond:<20s} mu={t.get('mu', 0):.4f} alpha={t.get('alpha', 0):.4f} "
                      f"params={n}")
    print(f"\nResults saved to {metrics_path}")


if __name__ == "__main__":
    main()
