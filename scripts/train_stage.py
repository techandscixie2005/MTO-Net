#!/usr/bin/env python3
"""Train MTO-Net for a specific stage (A/B/C).

Supports: locked dataset splits, checkpoint resume, stage initialization,
configurable activity gate, config YAML loading with CLI overrides,
spectral CSV loading for Stage B/C tasks.
"""
import argparse, json, os, sys, time, datetime
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.mto.mto_model import MTONet
from src.mto.training import Trainer, NormalizationStats
from src.mto.dataset_qm9s import (
    load_qm9s_raw, collate_batch, SpectralCollator,
    load_spectral_index, load_spectral_index_for_subset
)
from src.mto.data_splits import load_or_create_split, split_indices_for_seed
from src.mto.config_util import load_yaml_config, merge_configs, get_nested


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)


SPECTRAL_TASKS = {"ir", "raman", "uv"}


def get_stage_info(stage):
    """Return (task_list, spectral_tasks) for a stage."""
    stage_configs = {
        "stage_a": (["mu", "alpha"], []),
        "stage_b": (["mu", "alpha", "ir", "raman"], ["ir", "raman"]),
        "stage_c": (["mu", "alpha", "ir", "raman", "uv"], ["ir", "raman", "uv"]),
    }
    return stage_configs[stage]


class LazySubset(torch.utils.data.Dataset):
    def __init__(self, data_list, indices):
        self.data = data_list
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.data[self.indices[idx]]


def build_config(args):
    """Build effective config: defaults < YAML < CLI args."""
    config = {
        "model": {"feature_dim": 128, "maxl": 3, "num_block": 3, "rc": 5.0,
                   "mto_hidden_dim": 64, "readout_hidden_dim": 128},
        "train": {"epochs": 50, "lr": 1e-3, "batch_size": 8, "seed": 0,
                   "diversity_weight": 1e-3, "entropy_weight": 1e-3,
                   "theta_start": 0.5, "theta_end": 0.03, "theta_epochs": 30},
        "activity_gate": {"mode": "simple", "hidden_dim": 64},
        "data": {"train_frac": 0.8, "val_frac": 0.1, "split_seed": 0},
    }
    if args.config:
        yaml_cfg = load_yaml_config(args.config)
        config = merge_configs(config, yaml_cfg)
    # CLI overrides
    if args.feature_dim != 128:
        config["model"]["feature_dim"] = args.feature_dim
    if args.maxl != 3:
        config["model"]["maxl"] = args.maxl
    if args.num_block != 3:
        config["model"]["num_block"] = args.num_block
    if args.rc != 5.0:
        config["model"]["rc"] = args.rc
    if args.lr != 1e-3:
        config["train"]["lr"] = args.lr
    if args.batch_size != 8:
        config["train"]["batch_size"] = args.batch_size
    if args.epochs != 50:
        config["train"]["epochs"] = args.epochs
    config["train"]["seed"] = args.seed
    if hasattr(args, "activity_mode") and args.activity_mode is not None:
        config["activity_gate"]["mode"] = args.activity_mode
    if hasattr(args, "max_mols") and args.max_mols > 0:
        config["data"]["max_mols"] = args.max_mols
    return config


def main():
    parser = argparse.ArgumentParser(description="Train MTO-Net")
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
    parser.add_argument("--max-mols", type=int, default=0,
                        help="Limit total mols for smoke testing (0=all)")
    parser.add_argument("--split-file", default="outputs/splits/qm9s_split.json")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument("--init-from", default=None,
                        help="Initialize weights from stage checkpoint")
    parser.add_argument("--activity-mode", default=None,
                        choices=["none", "simple", "fermi_dirac"],
                        help="Activity gate mode (default: simple)")
    parser.add_argument("--save-cache", action="store_true",
                        help="Save MTO interpretability cache")
    parser.add_argument("--plot-mto", action="store_true",
                        help="Generate MTO diagnostic figures")
    parser.add_argument("--spectral-downsample", type=int, default=0,
                        help="Downsample spectral bins to N (0=use all)")
    parser.add_argument("--spectral-data-dir", default=None,
                        help="Dir with spectral CSV files (default: same as data-dir)")
    args = parser.parse_args()

    config = build_config(args)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("Device:", str(device))
    log("Stage:", args.stage)
    log("Activity gate mode:", config["activity_gate"]["mode"])

    # Load data
    pt_path = os.path.join(args.data_dir, "qm9s.pt")
    if not os.path.exists(pt_path):
        pt_path = os.path.join("data/qm9s", "qm9s.pt")
    log("Loading QM9S from", pt_path)
    raw_data = load_qm9s_raw(pt_path)
    log("Loaded", len(raw_data), "molecules")

    if args.max_mols > 0 and args.max_mols < len(raw_data):
        raw_data = raw_data[:args.max_mols]
        log("Limited to", len(raw_data), "molecules")

    n_total = len(raw_data)
    tasks, spectral_tasks = get_stage_info(args.stage)

    # Load spectral index for Stage B/C - only for needed mol_ids
    spectral_data_dir = args.spectral_data_dir or args.data_dir
    spectral_index = {}
    if spectral_tasks:
        if n_total < 10000:
            # Small dataset: load only needed mol IDs
            mol_ids_needed = set()
            for mol in raw_data[:n_total]:
                if hasattr(mol, "number"):
                    mol_ids_needed.add(int(mol.number))
            log("Loading spectral index for", len(mol_ids_needed), "mols...")
            spectral_index = load_spectral_index_for_subset(
                spectral_data_dir, spectral_tasks, mol_ids_needed)
        else:
            log("Loading spectral index for:", spectral_tasks)
            spectral_index = load_spectral_index(spectral_data_dir, spectral_tasks)
        for st in spectral_tasks:
            n_idx = len(spectral_index.get(st, {}))
            log(f"  {st}: {n_idx} molecules indexed")

    spectral_downsample = getattr(args, "spectral_downsample", 0) or 0

    # Build collator with spectral data if available
    if spectral_tasks and spectral_index:
        collator = SpectralCollator(spectral_index, spectral_tasks,
                                     downsample=spectral_downsample)
    else:
        collator = collate_batch

    # Locked split (now redundant - do simple split for smoke tests)
    if n_total < 1000:
        # Simple split for smoke tests
        n_train = max(1, int(n_total * 0.7))
        n_val = max(1, int(n_total * 0.15))
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_total))
    else:
        os.makedirs(os.path.dirname(args.split_file), exist_ok=True)
        split, was_created = load_or_create_split(
            args.split_file, n_total,
            train_frac=config["data"]["train_frac"],
            val_frac=config["data"]["val_frac"],
            seed=config["data"]["split_seed"],
        )
        if was_created:
            log("Created new split file:", args.split_file)
        else:
            log("Loaded existing split file:", args.split_file)
        train_indices, val_indices, test_indices = split_indices_for_seed(
            split, args.seed)
    log(f"Train: {len(train_indices)} Val: {len(val_indices)} Test: {len(test_indices)}")

    log("Tasks:", tasks)
    log("Spectral:", spectral_tasks if spectral_tasks else "none")

    train_loader = DataLoader(
        LazySubset(raw_data, train_indices),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_subset = val_indices[:min(len(val_indices), 2000)]
    val_loader = DataLoader(
        LazySubset(raw_data, val_subset),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collator,
    )

    # Normalization stats (including spectral)
    norm_stats = NormalizationStats()
    log("Computing normalization stats (sampling 2000)...")
    sample_indices = list(range(min(n_total, 2000)))
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
            log(f"  {task}: mean={s['mean']:.4f} std={s['std']:.4f}")

    # Compute spectral normalization stats from spectral index
    for st in spectral_tasks:
        idx = spectral_index.get(st, {})
        if idx:
            # Sample from spectral index
            spec_values = []
            for mol_i in sorted(idx.keys())[:500]:
                spec_values.append(torch.tensor(idx[mol_i], dtype=torch.float32))
            if spec_values:
                norm_stats.fit_tensors(st, spec_values)
                s = norm_stats.stats[st]
                log(f"  {st}: mean={s['mean']:.6f} std={s['std']:.6f}")

    # Build model - determine output dimensions
    spectral_bin_override = getattr(args, "spectral_downsample", 0) or 0
    out_dims = {
        "mu": 3, "alpha": 9,
        "ir": spectral_bin_override if spectral_bin_override > 0 else 3501,
        "raman": spectral_bin_override if spectral_bin_override > 0 else 3501,
        "uv": spectral_bin_override if spectral_bin_override > 0 else 701,
    }
    task_dict = {t: out_dims[t] for t in tasks}
    log("Output dimensions:", task_dict)

    model = MTONet(
        feature_dim=config["model"]["feature_dim"],
        mto_hidden_dim=config["model"]["mto_hidden_dim"],
        readout_hidden_dim=config["model"]["readout_hidden_dim"],
        tasks=task_dict,
        detanet_kwargs={
            "maxl": config["model"]["maxl"],
            "num_block": config["model"]["num_block"],
            "rc": config["model"]["rc"],
            "device": device,
        },
        use_activity_gate=(config["activity_gate"]["mode"] != "none"),
        activity_mode=config["activity_gate"]["mode"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model params: {n_params:,}")

    # Handle --resume and --init-from
    start_epoch = 0
    ckpt = None
    if args.resume:
        log("Resuming from:", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log(f"  Resumed at epoch {start_epoch}")

    if args.init_from:
        log("Initializing from:", args.init_from)
        init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(
            init_ckpt["model_state_dict"], strict=False)
        missing_head_keys = [k for k in missing
                           if any(h in k for h in ["readout.heads.", "heads."])]
        missing_backbone_keys = [k for k in missing
                                if k not in missing_head_keys]
        if missing_head_keys:
            log(f"  New heads (expected): {len(missing_head_keys)} keys")
        if missing_backbone_keys:
            log(f"  WARNING: missing backbone keys: {missing_backbone_keys}")
        if unexpected:
            log(f"  Unexpected keys: {len(unexpected)}")
        log("  Init-from completed")

    ckpt_dir = os.path.join(args.checkpoint_dir,
                            args.stage + "_seed" + str(args.seed))

    trainer = Trainer(
        model, device, tasks,
        lr=config["train"]["lr"],
        diversity_weight=config["train"]["diversity_weight"],
        entropy_weight=config["train"]["entropy_weight"],
        theta_start=config["train"]["theta_start"],
        theta_end=config["train"]["theta_end"],
        theta_epochs=config["train"]["theta_epochs"],
    )

    if args.resume and ckpt and "optimizer_state_dict" in ckpt:
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    log("Starting training...")
    history, best_path = trainer.fit(
        train_loader, val_loader, args.epochs,
        norm_stats=norm_stats, checkpoint_dir=ckpt_dir,
    )

    # Save metrics
    metrics_dir = "outputs/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    norm_stats.save(os.path.join(metrics_dir, args.stage + "_seed" + str(args.seed) + "_norm.json"))

    final_metrics = {
        "stage": args.stage, "seed": args.seed,
        "epochs": args.epochs, "params": n_params,
        "device": str(device),
        "activity_mode": config["activity_gate"]["mode"],
        "best_val_loss": history["val"][-1]["loss"] if history["val"] else None,
        "final_train_loss": history["train"][-1]["loss"] if history["train"] else None,
        "best_checkpoint": best_path,
        "split_file": args.split_file,
        "tasks": tasks,
        "spectral_tasks": spectral_tasks,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    metrics_path = os.path.join(metrics_dir, args.stage + "_seed" + str(args.seed) + "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)

    # Save MTO cache if requested
    if args.save_cache:
        log("Saving MTO cache...")
        from src.mto.analysis.mto_cache import MTOCache, build_mto_cache_entry
        cache_dir = "outputs/smoke/cache/"
        os.makedirs(cache_dir, exist_ok=True)
        cache = MTOCache(cache_dir, max_molecules=32)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 2:
                    break
                z = batch["z"].to(device)
                pos = batch["pos"].to(device)
                batch_idx = batch["batch"].to(device)
                out = model(z=z, pos=pos, batch=batch_idx, return_mto=True)
                for b in range(min(len(batch_idx.unique()), 4)):
                    entry = build_mto_cache_entry(
                        out, batch, mol_idx=b,
                        seed=args.seed, stage=args.stage,
                    )
                    cache.add_entry(entry)
        cache_path = cache.save(tag=f"seed{args.seed}")
        log(f"Cache saved: {cache_path}")
        final_metrics["cache_path"] = cache_path

    # Generate MTO figure if requested
    if args.plot_mto:
        log("Generating MTO diagnostic figure...")
        from src.mto.visualization import plot_mto_map
        from src.mto.molecule_builder import make_ethanol
        model.eval()
        _z, _pos, _batch = make_ethanol()
        _z = _z.to(device)
        _pos = _pos.to(device)
        _batch = _batch.to(device)
        with torch.no_grad():
            out = model(z=_z, pos=_pos, batch=_batch, return_mto=True)
        fig_dir = "outputs/figures/debug/smoke"
        os.makedirs(fig_dir, exist_ok=True)
        coeff = out["coeff"][0, 0, :len(_z)].abs()
        fig_path = plot_mto_map(_pos, _z, coeff,
                               mol_label="ethanol",
                               slot_idx=0,
                               K=int(out["K_per_mol"][0].item()),
                               save_path=os.path.join(fig_dir, "smoke_mto_map.png"))
        log(f"Figure saved: {fig_path}")
        final_metrics["figure_path"] = fig_path

    log("Best checkpoint:", best_path)
    log("Final val loss:", final_metrics["best_val_loss"])


if __name__ == "__main__":
    main()
