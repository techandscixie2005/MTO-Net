#!/usr/bin/env python3
"""Run baseline comparisons at smoke scale on QM9S subset.

Baselines compared:
1. Direct readout (no MTO, sum pool)
2. Attention pooling (learned weights)
3. No-sign MTO (softmax-only routing)
4. Fixed-K MTO (K=20 fixed)
5. Full MTO-Net (signed, valence-adaptive)

Saves metrics for comparison figure generation.
"""
import argparse, json, os, sys, time
import torch
import numpy as np

# proj_root = scripts/eval -> scripts -> repo root
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "third_party", "DetaNet"))

from src.mto.compat import *  # noqa: must be before any DetaNet imports
from src.mto.mto_model import MTONet
from src.mto.training import Trainer, NormalizationStats
from src.mto.dataset_qm9s import load_qm9s_raw, collate_batch
from src.mto.data_splits import load_or_create_split, split_indices_for_seed
from src.mto.config_util import load_yaml_config, merge_configs


class LazySubset(torch.utils.data.Dataset):
    def __init__(self, data_list, indices):
        self.data = data_list
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.data[self.indices[idx]]


def evaluate_model(model, loader, device, tasks):
    """Evaluate MAE for each task."""
    model.eval()
    errors = {t: [] for t in tasks}
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            pos = batch["pos"].to(device)
            batch_idx = batch["batch"].to(device)
            out = model(z=z, pos=pos, batch=batch_idx)
            for t in tasks:
                if t in out and t in batch:
                    diff = (out[t] - batch[t].to(device)).abs().mean().item()
                    errors[t].append(diff)
    return {t: float(np.mean(vs)) if vs else float("nan") for t, vs in errors.items()}


def run_baseline(name, model, tasks, train_subset, val_subset, test_subset,
                 norm_stats, device, args):
    """Train and evaluate a baseline model."""
    print(f"\n{'='*60}")
    print(f"  Baseline: {name}")
    print(f"{'='*60}")

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_batch)
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batch)
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batch)

    ckpt_dir = os.path.join(args.checkpoint_dir, name.replace(" ", "_"))
    os.makedirs(ckpt_dir, exist_ok=True)

    trainer = Trainer(model, device, tasks,
                      lr=args.lr, diversity_weight=0.0, entropy_weight=0.0)
    _, best_path = trainer.fit(train_loader, val_loader, args.epochs,
                               norm_stats=norm_stats, checkpoint_dir=ckpt_dir)

    # Reload best and evaluate
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_metrics = evaluate_model(model, val_loader, device, tasks)
    test_metrics = evaluate_model(model, test_loader, device, tasks)

    print(f"  Val:  {val_metrics}")
    print(f"  Test: {test_metrics}")

    return {
        "name": name,
        "val": val_metrics,
        "test": test_metrics,
        "checkpoint": best_path,
        "params": sum(p.numel() for p in model.parameters()),
    }


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

    split = load_or_create_split(
        os.path.join("outputs", "splits", "qm9s_split_smoke.json"), int(n),
        train_frac=0.7, val_frac=0.15)
    train_idx, val_idx, test_idx = split_indices_for_seed(split, 0)

    train_subset = LazySubset(data, train_idx)
    val_subset = LazySubset(data, val_idx)
    test_subset = LazySubset(data, test_idx)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Norm stats (tiny subset)
    norm_stats = NormalizationStats()
    sample_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=min(64, len(train_idx)), shuffle=True,
        collate_fn=collate_batch)
    for batch in sample_loader:
        norm_stats.update(batch)
        break
    norm_stats.finalize()
    print(f"Norm: mu_mean={norm_stats.mean.get('mu', 0):.4f}, "
          f"alpha_mean={norm_stats.mean.get('alpha', 0):.4f}")

    results = []

    # 1. Full MTO-Net
    print("\n[1/5] Full MTO-Net")
    model_full = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                        readout_hidden_dim=64, tasks=tasks,
                        detanet_kwargs={"maxl": args.maxl, "num_block": args.num_block,
                                        "rc": args.rc, "device": device},
                        use_activity_gate=True, activity_mode="simple")
    model_full = model_full.to(device)
    r = run_baseline("full_mto", model_full, tasks, train_subset, val_subset,
                     test_subset, norm_stats, device, args)
    results.append(r)

    # 2. MTO without sign (softmax-only routing)
    print("\n[2/5] No-sign MTO")
    model_nosign = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                          readout_hidden_dim=64, tasks=tasks,
                          detanet_kwargs={"maxl": args.maxl, "num_block": args.num_block,
                                          "rc": args.rc, "device": device},
                          use_activity_gate=True, activity_mode="simple")
    # Override sign MLPs to return all-ones
    def _zero_sign(model):
        for m in [model.mto.sign_mlp_l0, model.mto.sign_mlp_l1, model.mto.sign_mlp_l2]:
            for layer in m:
                if hasattr(layer, 'weight'):
                    nn.init.constant_(layer.weight, 0)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    import torch.nn as nn
    _zero_sign(model_nosign)
    model_nosign = model_nosign.to(device)
    r = run_baseline("no_sign_mto", model_nosign, tasks, train_subset, val_subset,
                     test_subset, norm_stats, device, args)
    results.append(r)

    # 3. Fixed-K MTO (K=20)
    print("\n[3/5] Fixed-K MTO")
    model_fixedk = MTONet(feature_dim=args.feature_dim, mto_hidden_dim=32,
                          readout_hidden_dim=64, tasks=tasks,
                          detanet_kwargs={"maxl": args.maxl, "num_block": args.num_block,
                                          "rc": args.rc, "device": device},
                          use_activity_gate=True, activity_mode="simple")
    # Force K=20 for all molecules
    model_fixedk.mto._original_forward = model_fixedk.mto.forward
    def fixed_k_forward(atom_features, z, batch=None, theta=0.5):
        # Monkey-patch to force K=20
        from src.mto.valence import molecular_valence_electrons
        n_atoms = atom_features.shape[0]
        if batch is None:
            batch = torch.zeros(n_atoms, dtype=torch.long, device=atom_features.device)
        max_K = 20
        B = int(batch.max().item()) + 1
        K_vec = torch.full((B,), max_K, dtype=torch.long, device=atom_features.device)
        K_max = max_K
        mask = torch.ones(B, K_max, dtype=torch.bool, device=atom_features.device)

        # Use original logic but with fixed K_max
        from src.mto.mto_module import ValenceAdaptiveMTO
        original = model_fixedk.mto
        # Build routing
        atom_type_ids = z.clamp(0, 19)
        type_emb = original.atom_type_emb(atom_type_ids)
        routing_logits = []
        coeff_list = []
        O = torch.zeros(B, K_max, original.feature_dim, device=atom_features.device)
        for b in range(B):
            mask_b = (batch == b)
            af_b = atom_features[mask_b]
            te_b = type_emb[mask_b]
            n_a = af_b.shape[0]
            K_b = K_vec[b].item()
            # Use standard routing but iterate slots
            O_b = torch.zeros(K_b, original.feature_dim, device=atom_features.device)
            c_b = torch.zeros(K_b, n_a, device=atom_features.device)
            for k in range(K_b):
                slot_k = original.slot_emb(torch.tensor([k % 128], device=atom_features.device))
                se_k = slot_k.expand(n_a, -1)
                concat = torch.cat([af_b, te_b, se_k], dim=-1)
                e = original.route_mlp(concat).squeeze(-1)
                a = F.softmax(e, dim=-1)
                s = torch.tanh(original.sign_mlp_l0(concat).squeeze(-1))
                c = a * s / (torch.abs(a * s).sum() + 1e-6)
                c_b[k] = c
                O_b[k] = (c.unsqueeze(-1) * af_b).sum(dim=0)
            O[b, :K_b] = O_b
            coeff_list.append(c_b)

        if original.use_activity_gate:
            O_flat = O.view(B * K_max, -1)
            activity = original.activity_gate(O_flat).view(B, K_max)
            O = O * activity.unsqueeze(-1)
        else:
            activity = torch.ones(B, K_max, device=atom_features.device)

        return {
            "O": O, "mask": mask, "coeff": coeff_list,
            "atom_mask": mask.new_ones(n_atoms).bool(),
            "K_per_mol": K_vec, "activity": activity,
            "routing_logits": routing_logits,
        }
    model_fixedk.mto.forward = fixed_k_forward
    model_fixedk = model_fixedk.to(device)
    r = run_baseline("fixed_k_mto", model_fixedk, tasks, train_subset, val_subset,
                     test_subset, norm_stats, device, args)
    results.append(r)

    # 4. Direct readout (no MTO, sum pool)
    print("\n[4/5] Direct readout (no MTO)")
    from src.mto.baselines import DirectReadoutBaseline
    model_direct = DirectReadoutBaseline(
        model_full.backbone, tasks, hidden_dim=64)
    model_direct = model_direct.to(device)
    r = run_baseline("direct_readout", model_direct, tasks, train_subset, val_subset,
                     test_subset, norm_stats, device, args)
    results.append(r)

    # 5. Attention pooling
    print("\n[5/5] Attention pooling")
    from src.mto.baselines import AttentionPoolingBaseline
    model_attn = AttentionPoolingBaseline(
        model_full.backbone, tasks, hidden_dim=64)
    model_attn = model_attn.to(device)
    r = run_baseline("attention_pooling", model_attn, tasks, train_subset, val_subset,
                     test_subset, norm_stats, device, args)
    results.append(r)

    # Save results
    os.makedirs(args.metrics_dir, exist_ok=True)
    metrics_path = os.path.join(args.metrics_dir, "baseline_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {metrics_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25s} {'mu MAE':>10s} {'alpha MAE':>10s} {'Params':>10s}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<25s} {r['test'].get('mu', float('nan')):>10.4f} "
              f"{r['test'].get('alpha', float('nan')):>10.4f} {r['params']:>10d}")
    print("=" * 70)


if __name__ == "__main__":
    main()
