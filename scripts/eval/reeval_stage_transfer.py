#!/usr/bin/env python3
"""Re-evaluate existing stage transfer checkpoints with normalized metrics and subspace sim."""
import json, os, sys, torch, numpy as np

_proj_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "third_party", "DetaNet"))

from src.mto.compat import *
from src.mto.mto_model import MTONet
from src.mto.dataset_qm9s import load_qm9s_raw, collate_batch
from src.mto.data_splits import generate_split, split_indices_for_seed


class LazySubset(torch.utils.data.Dataset):
    def __init__(self, data_list, indices):
        self.data = data_list; self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.data[self.indices[idx]]


@torch.no_grad()
def evaluate(model, loader, device, task_list, mu_mean, mu_std, alpha_mean, alpha_std):
    model.eval()
    raw_errs = {t: [] for t in task_list}
    norm_errs = {t: [] for t in task_list}
    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        batch_idx = batch["batch"].to(device)
        out = model(z=z, pos=pos, batch=batch_idx)
        for t in task_list:
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


@torch.no_grad()
def compute_mto_subspace(model, loader, device, top_r=5):
    """Extract MTO coefficient subspace by averaging over atoms per molecule.
    coeff has shape (B, K_max, N_atoms) — K varies by molecule, N_atoms varies by batch.
    Returns U[:, :top_r] from SVD of (total_mols, global_K_max) matrix."""
    coeffs_list = []
    for batch in loader:
        z = batch["z"].to(device)
        pos = batch["pos"].to(device)
        batch_idx = batch["batch"].to(device)
        out = model(z=z, pos=pos, batch=batch_idx, return_mto=True)
        if "coeff" in out and out["coeff"] is not None:
            c = out["coeff"].detach().cpu()  # (B, K_max, N_atoms)
            # Average over atoms -> (B, K_max)
            c_mean = c.mean(dim=-1)
            coeffs_list.append(c_mean)
    if not coeffs_list:
        return None
    # Pad K dimension to global max_K across all batches
    max_K = max(c.shape[1] for c in coeffs_list)
    padded = []
    for c in coeffs_list:
        if c.shape[1] < max_K:
            c = torch.nn.functional.pad(c, (0, max_K - c.shape[1]))
        padded.append(c)
    all_coeffs = torch.cat(padded, dim=0)  # (total_mols, max_K)
    U, _, _ = torch.svd(all_coeffs.float())
    return U[:, :top_r]


def subspace_similarity(U1, U2):
    if U1 is None or U2 is None:
        return None
    M = U1.T @ U2
    _, s, _ = torch.svd(M)
    return float(s.mean())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/qm9s/subset_medium")
    parser.add_argument("--checkpoint-dir", default="outputs/stage_transfer")
    parser.add_argument("--output", default="outputs/stage_transfer/stage_transfer_results.json")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--maxl", type=int, default=3)
    parser.add_argument("--num-block", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dk = {"maxl": args.maxl, "num_block": args.num_block, "rc": 5.0, "device": device}

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

        seed_results = {"seed": seed, "conditions": {}}
        models_by_cond = {}

        for cond_name, cond_tasks in conditions.items():
            ckpt_path = os.path.join(args.checkpoint_dir, f"seed{seed}", cond_name, "best.pt")
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {cond_name}: checkpoint not found at {ckpt_path}")
                continue

            print(f"\n  --- {cond_name} (tasks: {list(cond_tasks.keys())}) ---")
            model = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                          readout_hidden_dim=64, tasks=cond_tasks, detanet_kwargs=dk,
                          use_activity_gate=True, activity_mode="simple").to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            models_by_cond[cond_name] = model

            raw_test, norm_test = evaluate(model, test_loader, device, eval_tasks,
                                           mu_mean, mu_std, alpha_mean, alpha_std)
            seed_results["conditions"][cond_name] = {
                "train_tasks": list(cond_tasks.keys()),
                "test_raw": raw_test, "test_norm": norm_test,
            }
            print(f"  Test raw: {raw_test} | norm: {norm_test}")

        # MTO subspace similarity
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

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 90)
    print("  STAGE TRANSFER RE-EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Seed':<6s} {'Condition':<14s} {'mu_raw':>8s} {'mu_norm':>8s} {'a_raw':>8s} {'a_norm':>8s} {'S_sub':>10s}")
    print("-" * 80)
    for sr in all_results:
        s = sr["seed"]
        for cn in conditions:
            tr = sr["conditions"].get(cn, {}).get("test_raw", {})
            tn = sr["conditions"].get(cn, {}).get("test_norm", {})
            sub = sr.get("subspace_sim", {}).get("mu_only_vs_alpha_only")
            sub_str = f"{sub:.4f}" if sub is not None else "N/A"
            print(f"{s:<6d} {cn:<14s} {tr.get('mu', 0):>8.4f} {tn.get('mu', 0):>8.4f} "
                  f"{tr.get('alpha', 0):>8.4f} {tn.get('alpha', 0):>8.4f} {sub_str:>10s}")
    print("=" * 90)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
