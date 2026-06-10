#!/usr/bin/env python3
"""Re-evaluate existing ablation checkpoints with proper normalized metrics."""
import json, os, sys, torch, numpy as np

_proj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "third_party", "DetaNet"))

from src.mto.compat import *
from src.mto.mto_model import MTONet
from src.mto.dataset_qm9s import load_qm9s_raw, collate_batch
from src.mto.data_splits import generate_split, split_indices_for_seed
from src.mto.baselines import FixedKTokenBaseline, DirectReadoutBaseline, AttentionPoolingBaseline


class LazySubset(torch.utils.data.Dataset):
    def __init__(self, data_list, indices):
        self.data = data_list; self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.data[self.indices[idx]]


@torch.no_grad()
def evaluate(model, loader, device, tasks, mu_mean, mu_std, alpha_mean, alpha_std):
    model.eval()
    raw_errs = {t: [] for t in tasks}
    norm_errs = {t: [] for t in tasks}
    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        batch_idx = batch["batch"].to(device)
        out = model(z=z, pos=pos, batch=batch_idx)
        for t in tasks:
            if t in out and t in batch:
                target = batch[t].to(device)
                raw_errs[t].append((out[t] - target).abs().mean().item())
                if t == "mu":
                    tn = (target - mu_mean) / max(mu_std, 1e-6)
                    pn = (out[t] - mu_mean) / max(mu_std, 1e-6)
                elif t == "alpha":
                    tn = (target - alpha_mean) / max(alpha_std, 1e-6)
                    pn = (out[t] - alpha_mean) / max(alpha_std, 1e-6)
                else:
                    tn = target; pn = out[t]
                norm_errs[t].append((pn - tn).abs().mean().item())
    raw = {t: float(np.mean(vs)) if vs else float("nan") for t, vs in raw_errs.items()}
    norm = {t: float(np.mean(vs)) if vs else float("nan") for t, vs in norm_errs.items()}
    return raw, norm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/qm9s/subset_medium")
    parser.add_argument("--checkpoint-dir", default="outputs/baselines")
    parser.add_argument("--output", default="outputs/metrics/baselines/baseline_comparison.json")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--maxl", type=int, default=3)
    parser.add_argument("--num-block", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = {"mu": 3, "alpha": 9}
    dk = {"maxl": args.maxl, "num_block": args.num_block, "rc": 5.0, "device": device}

    pt_path = os.path.join(args.data_dir, "qm9s.pt") if os.path.isdir(args.data_dir) else args.data_dir
    data = load_qm9s_raw(pt_path)
    print(f"Molecules: {len(data)}")

    all_results = []
    for seed in [0, 1, 2]:
        print(f"\n{'='*60}\n  Re-evaluating seed={seed}\n{'='*60}")
        split = generate_split(len(data), train_frac=0.7, val_frac=0.15, seed=seed)
        train_idx, val_idx, test_idx = split_indices_for_seed(split, seed)
        test_sub = LazySubset(data, test_idx)
        test_loader = torch.utils.data.DataLoader(
            test_sub, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        # Norm stats from train
        train_sub = LazySubset(data, train_idx)
        mu_vals, alpha_vals = [], []
        for i in range(min(200, len(train_idx))):
            mol = train_sub[i]
            mu_vals.append(mol.dipole.clone())
            alpha_vals.append(mol.polar.reshape(-1).clone())
        mu_all = torch.cat([t.reshape(-1) for t in mu_vals])
        alpha_all = torch.cat([t.reshape(-1) for t in alpha_vals])
        mu_mean = float(mu_all.mean()); mu_std = max(float(mu_all.std()), 1e-6)
        alpha_mean = float(alpha_all.mean()); alpha_std = max(float(alpha_all.std()), 1e-6)
        print(f"Norm: mu({mu_mean:.3f}+/-{mu_std:.3f}) alpha({alpha_mean:.3f}+/-{alpha_std:.3f})")

        seed_dir = os.path.join(args.checkpoint_dir, f"seed{seed}")

        for method_key, display_name in [
            ("full_mto", "full_mto"), ("no_sign_mto", "no_sign_mto"),
            ("fixed_k_mto", "fixed_k_mto"), ("direct_readout", "direct_readout"),
            ("attention_pooling", "attention_pooling"),
        ]:
            ckpt_path = os.path.join(seed_dir, method_key, "best.pt")
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {display_name}: checkpoint not found at {ckpt_path}")
                continue

            print(f"\n  --- {display_name} ---")
            # Build matching model
            if method_key == "fixed_k_mto":
                # Build full MTO first for backbone, then FixedK
                m_full = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                                readout_hidden_dim=64, tasks=tasks, detanet_kwargs=dk,
                                use_activity_gate=True, activity_mode="simple").to(device)
                ckpt = torch.load(os.path.join(seed_dir, "full_mto", "best.pt"),
                                  map_location=device, weights_only=False)
                m_full.load_state_dict(ckpt["model_state_dict"], strict=False)
                model = FixedKTokenBaseline(m_full.backbone, tasks, K_fixed=20,
                                            feature_dim=args.feature_dim, hidden_dim=32).to(device)
            elif method_key in ("direct_readout", "attention_pooling"):
                m_full = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                                readout_hidden_dim=64, tasks=tasks, detanet_kwargs=dk,
                                use_activity_gate=True, activity_mode="simple").to(device)
                ckpt = torch.load(os.path.join(seed_dir, "full_mto", "best.pt"),
                                  map_location=device, weights_only=False)
                m_full.load_state_dict(ckpt["model_state_dict"], strict=False)
                if method_key == "direct_readout":
                    model = DirectReadoutBaseline(m_full.backbone, tasks, hidden_dim=64).to(device)
                else:
                    model = AttentionPoolingBaseline(m_full.backbone, tasks, hidden_dim=64).to(device)
            else:
                model = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                               readout_hidden_dim=64, tasks=tasks, detanet_kwargs=dk,
                               use_activity_gate=True, activity_mode="simple").to(device)

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

            raw_test, norm_test = evaluate(model, test_loader, device, tasks,
                                           mu_mean, mu_std, alpha_mean, alpha_std)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Test raw: {raw_test} | norm: {norm_test} | params: {n_params}")

            all_results.append({
                "seed": seed, "method": display_name,
                "test_raw": raw_test, "test_norm": norm_test,
                "params": n_params,
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 90)
    print("  ABLATION RE-EVALUATION SUMMARY (test set)")
    print("=" * 90)
    print(f"{'Method':<22s} {'Seed':>5s} {'mu_raw':>8s} {'mu_norm':>8s} {'a_raw':>8s} {'a_norm':>8s}")
    print("-" * 66)
    for r in all_results:
        print(f"{r['method']:<22s} {r['seed']:>5d} "
              f"{r['test_raw'].get('mu', 0):>8.4f} {r['test_norm'].get('mu', 0):>8.4f} "
              f"{r['test_raw'].get('alpha', 0):>8.4f} {r['test_norm'].get('alpha', 0):>8.4f}")
    print("=" * 90)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
